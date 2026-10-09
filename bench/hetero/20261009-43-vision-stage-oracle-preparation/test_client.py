"""Host-only metadata/hash fixtures; no native process, model or Core."""
import copy, hashlib, importlib.util, io, json, struct, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("vision_stage_test_client",HERE/"client.py")
client=importlib.util.module_from_spec(spec);sys.modules[spec.name]=client;spec.loader.exec_module(client)


class StageClientFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="vision-stage-client-fixture-")
        self.root=Path(self.temp.name)
        self.c=client.document(client.CONTRACT)
    def tearDown(self):self.temp.cleanup()

    def synthetic_manifest(self):
        prefix=self.root/"SYNTHETIC";prefix.mkdir()
        fixture={"name":"SYNTHETIC","path":str(self.root/"SYNTHETIC.png"),"width":96,"height":96}
        stages=[];n=36;total=0
        for name,selector in self.c["selectors"].items():
            features,rows=(3456,n) if name=="qkv.0" else ((4608,n//4) if name=="merged" else (1152,n))
            ne=[features,rows,1,1];nb=[4,features*4,features*rows*4,features*rows*4]
            selected_ne,selected_nb=([72,16,n,1],[4,288,13824,13824*n]) if name=="qkv.0" else (ne,nb)
            raw=b"\0"*(features*rows*4);path=prefix/(name+".f32");path.write_bytes(raw);total+=len(raw)
            stages.append({"stage":name,"selector":selector,"actual_dtype":"F32","type_id":0,"ne":ne,"nb":nb,
                "selector_ne":selected_ne,"selector_nb":selected_nb,"path":str(path),"bytes":len(raw),
                "raw_sha256":hashlib.sha256(raw).hexdigest()})
        manifest={"schema_version":1,"status":"stages_captured","source_revision":self.c["source_revision"],
            "image":fixture["path"],"sve_output":str(self.root/"SYNTHETIC.sve"),"capture_number":1,
            "capture_bytes":total,"image_total_bytes":total,"image_byte_cap":client.CAP,"stages":stages}
        (prefix/"manifest.json").write_text(json.dumps(manifest))
        return prefix,fixture,manifest

    def test_default_is_plan_only_and_run_requires_root_authorization_before_guard_or_inputs(self):
        output=io.StringIO()
        with mock.patch.object(client,"native_run",side_effect=AssertionError("runtime called")),mock.patch("sys.stdout",output):
            self.assertEqual(client.main([]),0)
        plan=json.loads(output.getvalue());self.assertEqual(plan["native_commands"],20)
        self.assertFalse(plan["helper_started"] or plan["gguf_read"] or plan["core_created"])
        with mock.patch.object(client,"guard_api",side_effect=AssertionError("guard resolved before authorization")):
            with self.assertRaisesRegex(ValueError,"not authorized"):client.native_run(self.c)
        c=copy.deepcopy(self.c);c["execution_authorized"]=True
        c["owned_process_guard"]["reviewed_binding"]=None
        with self.assertRaisesRegex(ValueError,"root-bound"):client.guard_api(c)

    def test_native_direct_root_requires_exact_full_argv_without_a_cpython_child(self):
        tree=types.SimpleNamespace(root_record={"command_line":list(self.c["helper_argv"]),"pid":100})
        self.assertTrue(client.verify_native_root(tree,self.c)["full_argv_verified"])
        tree.root_record["command_line"][6]="11"
        with self.assertRaisesRegex(ValueError,"full argv differs"):client.verify_native_root(tree,self.c)

    def test_manifest_all_axes_hashes_and_budget_with_synthetic_f32_only(self):
        prefix,fixture,manifest=self.synthetic_manifest()
        got,total=client.validate_manifest(prefix,fixture,self.c,1,0,self.root/"SYNTHETIC.sve")
        self.assertEqual(total,2156544);self.assertEqual(len(got["stages"]),11)
        for mutation in ("missing","duplicate","stride","dtype","hash","budget","qview","sve"):
            bad=copy.deepcopy(manifest)
            if mutation=="missing":bad["stages"].pop()
            elif mutation=="duplicate":bad["stages"][1]=bad["stages"][0]
            elif mutation=="stride":bad["stages"][0]["nb"][1]+=4
            elif mutation=="dtype":bad["stages"][0]["actual_dtype"]="BF16"
            elif mutation=="hash":bad["stages"][0]["raw_sha256"]="a"*64
            elif mutation=="budget":bad["image_byte_cap"]+=1
            elif mutation=="qview":next(s for s in bad["stages"] if s["stage"]=="qkv.0")["selector_nb"][2]=4608
            else:bad["sve_output"]=str(self.root/"OTHER.sve")
            (prefix/"manifest.json").write_text(json.dumps(bad))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                client.validate_manifest(prefix,fixture,self.c,1,0,self.root/"SYNTHETIC.sve")

    def test_native_reference_sha_is_exact_not_tolerance_or_new_reference(self):
        expected={"sha256":"a"*64,"n_tokens":9,"nx":3,"ny":3,"n_embd":2560,"size_bytes":92180}
        client.require_frozen_sve(dict(expected),expected)
        with self.assertRaisesRegex(ValueError,"frozen run23"):
            client.require_frozen_sve(dict(expected,sha256="b"*64),expected)

    def test_ov_alias_projection_changes_are_explicit_and_native_reference_never_replaced(self):
        arms=[]
        for policy in ("graph_f32","cpu_vec_dot_rounding"):
            images=[]
            for fixture in self.c["fixtures"]:
                old=fixture["frozen_ov_decoded_sha256"][policy]
                images.append({"fixture":fixture["name"],"formal":[{"iteration":i,"output":{"decoded_sha256":old}} for i in (1,2,3)]})
            arms.append({"policy":policy,"images":images})
        receipt={"arms":arms}
        self.assertFalse(client.compare_ov_fingerprints(receipt,self.c)["projection_changed_from_frozen_run31"])
        arms[0]["images"][0]["formal"][0]["output"]["decoded_sha256"]="0"*64
        result=client.compare_ov_fingerprints(receipt,self.c)
        self.assertTrue(result["projection_changed_from_frozen_run31"])
        self.assertFalse(result["native_reference_replaced"] or result["numerical_gate_relaxed"])
        self.assertEqual(sum(row["changed"] for row in result["rows"]),1)


if __name__=="__main__":unittest.main()
