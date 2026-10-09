"""Host-only deadline/ownership fixtures; no worker, model, NumPy or Core."""
import copy, importlib.util, json, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("vision_ov_fixture_driver",HERE/"ov_driver.py")
driver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=driver;spec.loader.exec_module(driver)


class OvDriverFixtures(unittest.TestCase):
    def test_outer_controller_argv0_windows_normalization_keeps_exact_ordered_tail(self):
        expected=["E:/Strata-Hetero-data/venv/Scripts/python.exe","-u","-B","C:/owned/driver.py","--run"]
        row={"command_line":["E:\\Strata-Hetero-data\\venv\\Scripts\\python.exe",*expected[1:]]}
        self.assertTrue(driver.verify_launcher_argv(row,expected)["ordered_argv_tail_exact"])
        row["command_line"][1]="-B"
        with self.assertRaisesRegex(ValueError,"ordered argv differs"):driver.verify_launcher_argv(row,expected)

    def test_deadline_is_1800_and_wait_is_bounded_without_real_sleep(self):
        self.assertEqual(driver.WORKER_TIMEOUT_SECONDS,1800)
        owner=types.SimpleNamespace(returncode=None);monitor=mock.Mock();elapsed=[0.]
        with self.assertRaisesRegex(TimeoutError,"1800-second"):
            driver.wait_worker(owner,monitor,1.,clock=lambda:elapsed[0],pause=lambda dt:elapsed.__setitem__(0,elapsed[0]+dt))
        self.assertEqual(elapsed[0],1.);self.assertGreater(monitor.require.call_count,1)
        owner.returncode=0
        self.assertEqual(driver.wait_worker(owner,monitor,1.,clock=lambda:2.,pause=lambda dt:self.fail("slept after exit")),0)

    def test_timeout_reaps_bound_descendants_then_launcher_and_preserves_partial(self):
        with tempfile.TemporaryDirectory(prefix="vision-ov-deadline-fixture-") as folder:
            c=copy.deepcopy(driver.client.document(driver.client.CONTRACT));c["execution_authorized"]=True
            c["ov_diagnostic_root"]=str(Path(folder)/"synthetic-output")
            events=[];owner=types.SimpleNamespace(returncode=None,pid=123)
            def close_owner():events.append("launcher");owner.returncode=-15
            owner.close=close_owner
            tree=types.SimpleNamespace(await_actual_child=lambda **kw:{"synthetic_bound_child":True},
                cleanup_descendants=lambda:events.append("descendants") or {"remaining_owned_descendants":[]})
            monitor=types.SimpleNamespace(start=lambda:None,require=lambda:None,attach=lambda *args:None,
                close=lambda:events.append("monitor"),failed=None)
            shared=types.SimpleNamespace(ResourceMonitor=lambda *args,**kw:monitor,OwnedServiceTree=lambda *args:tree,
                utc=lambda:"2026-10-09T00:00:00+00:00",_process_record=lambda proc:{"synthetic":True},BenchError=RuntimeError)
            def synthetic_popen(*args,**kw):
                output=Path(c["ov_diagnostic_root"]);output.mkdir();(output/"partial.marker").write_text("SYNTHETIC")
                return types.SimpleNamespace(pid=123)
            native={"status":"native_stage_fingerprint_pass","results":[{}]*20}
            with mock.patch.object(driver.client,"inspect"),mock.patch.object(driver.client,"guard_api",return_value=shared),\
                 mock.patch.object(driver.client,"document",return_value=native),mock.patch.object(driver.client,"DirectOwner",return_value=owner),\
                 mock.patch.object(driver.subprocess,"Popen",side_effect=synthetic_popen),\
                 mock.patch.object(driver,"wait_worker",side_effect=TimeoutError("bounded 1800-second fixture timeout")):
                result=driver.execute(c)
            self.assertEqual(result["status"],"failed_preserved");self.assertTrue(result["owned_tree_terminal"])
            self.assertEqual(events,["descendants","launcher","monitor"])
            self.assertTrue((Path(c["ov_diagnostic_root"])/"partial.marker").is_file())
            receipt=Path(c["ov_diagnostic_root"]+"-controller")/"controller-receipt.json"
            self.assertIn("TimeoutError",json.loads(receipt.read_text())["error"])


if __name__=="__main__":unittest.main()
