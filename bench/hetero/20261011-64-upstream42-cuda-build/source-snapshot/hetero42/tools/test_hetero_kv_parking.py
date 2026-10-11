"""Pure bounded fixtures for the plan/quality/evidence layer; never launches a model."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

from tools import hetero_kv_parking as kv


class StrictContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.contract_sha, cls.configs = kv.load_contract()

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        with self.assertRaises(kv.ParkingError): kv.strict_json_loads('{"a":1,"a":2}')
        with self.assertRaises(kv.ParkingError): kv.strict_json_loads('{"a":{"b":1,"b":2}}')
        with self.assertRaises(kv.ParkingError): kv.strict_json_loads('{"a":NaN}')

    def test_plan_is_default_off_and_does_not_create_snapshots(self):
        plan = kv.build_plan(self.contract, self.contract_sha, self.configs)
        self.assertEqual(plan["status"], "plan_only")
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["snapshot_directory_created"])
        self.assertFalse(plan["snapshot_written"])
        self.assertFalse(plan["api_or_ov_used"])
        self.assertFalse(plan["gpu_used"])
        self.assertFalse(plan["all_storage_tiers_complete"])
        self.assertIn("runtime_binary_not_bound", plan["blockers"])

    def test_ram_and_disk_config_controls_are_separate(self):
        self.assertEqual(self.configs["ram-pair"]["cache_mib"], 4096)
        self.assertEqual(self.configs["disk-pair"]["cache_mib"], 0)
        self.assertIsNone(self.configs["ram-pair"].get("slot_save_path"))
        self.assertNotEqual(self.contract["snapshot_paths"]["directory"], "")

    def test_current_contract_cannot_authorize_or_import_an_adapter(self):
        with self.assertRaises(kv.ParkingError):
            kv.verify_execution_authorization(self.contract, self.contract_sha, {}, Path("missing-adapter.py"))


class RequestReferenceTests(StrictContractTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.requests_doc = make_request_manifest(cls.contract, cls.contract_sha)
        cls.requests_sha = kv.sha256_bytes(kv.canonical_json_bytes(cls.requests_doc))
        cls.requests = kv.validate_request_manifest(cls.requests_doc, cls.contract, cls.contract_sha)
        cls.references_doc = make_reference_manifest(cls.contract, cls.contract_sha, cls.requests_doc, cls.requests_sha)
        cls.references = kv.validate_reference_manifest(cls.references_doc, cls.requests, cls.contract,
                                                        cls.contract_sha, cls.requests_sha)

    def test_exact_contract_history_and_own_reference_pass(self):
        actual = completion_bundle(self.references_doc)
        result = kv.validate_arm_completions(actual, self.requests, self.references, self.contract)
        self.assertEqual(result["status"], "pass")
        self.assertIn("only to their own", result["reference_scope"])

    def test_reencoded_ids_are_not_used_as_emitted_ids(self):
        actual = completion_bundle(self.references_doc)
        actual["A"][1]["canonical_reencoded_token_ids"] = [999999]
        self.assertEqual(kv.validate_arm_completions(actual, self.requests, self.references, self.contract)["status"], "pass")

    def test_cross_conversation_reference_is_rejected(self):
        actual = completion_bundle(self.references_doc)
        actual["B"][1]["emitted_token_ids"] = list(self.references["A"]["formal1"]["emitted_token_ids"])
        with self.assertRaisesRegex(kv.ParkingError, "own uninterrupted reference"):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)

    def test_exact_generation_cap_is_enforced(self):
        doc = copy.deepcopy(self.requests_doc)
        doc["conversations"]["A"]["turns"][1]["max_tokens"] = 127
        with self.assertRaisesRegex(kv.ParkingError, "generation cap"):
            kv.validate_request_manifest(doc, self.contract, self.contract_sha)

    def test_exact_prompt_bytes_and_token_ids_are_bound(self):
        doc = copy.deepcopy(self.requests_doc)
        doc["conversations"]["A"]["turns"][2]["prompt_utf8"] += "!"
        with self.assertRaisesRegex(kv.ParkingError, "prompt bytes/hash"):
            kv.validate_request_manifest(doc, self.contract, self.contract_sha)

    def test_duplicate_output_keys_are_rejected_even_if_values_match(self):
        actual = completion_bundle(self.references_doc)
        actual["A"][1]["raw_output_utf8"] = '{"conversation":"A","conversation":"A"}'
        with self.assertRaisesRegex(kv.ParkingError, "duplicate JSON key"):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)

    def test_nonfinite_output_json_is_rejected(self):
        actual = completion_bundle(self.references_doc)
        actual["A"][1]["raw_output_utf8"] = '{"x":Infinity}'
        with self.assertRaises(kv.ParkingError):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)

    def test_legacy_v1_artifacts_cannot_claim_a_checkpoint_tail(self):
        actual = completion_bundle(self.references_doc)
        actual["B"][2]["checkpoint_tail_sha256"] = "a" * 64
        with self.assertRaisesRegex(kv.ParkingError, "checkpoint-tail"):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)


class TierEvidenceTests(StrictContractTests):
    def test_ram_evidence_requires_both_chats_reuse_and_byte_cap(self):
        result = kv.validate_ram_evidence(ram_evidence(self.contract), self.contract)
        self.assertEqual(result["tier"], "ram-hot")
        self.assertLessEqual(result["max_cache_bytes"], self.contract["resource_budget"]["ram_cache_cap_bytes"])

    def test_ram_evidence_rejects_over_cap_or_missing_restores(self):
        bad = ram_evidence(self.contract)
        bad["events"][0]["cache_bytes"] = self.contract["resource_budget"]["ram_cache_cap_bytes"] + 1
        with self.assertRaisesRegex(kv.ParkingError, "byte cap"):
            kv.validate_ram_evidence(bad, self.contract)
        bad = ram_evidence(self.contract)
        bad["events"] = [x for x in bad["events"] if not (x["conversation"] == "B" and x["action"] == "restore")]
        with self.assertRaisesRegex(kv.ParkingError, "three actual park/restore"):
            kv.validate_ram_evidence(bad, self.contract)

    def test_disk_seed_and_checkpoint_evidence_are_distinct_from_ram(self):
        evidence, tokens, tails = disk_evidence(self.contract)
        result = kv.validate_disk_evidence(evidence, self.contract, tokens)
        self.assertEqual(result["tier"], "disk-parked-cold")
        with self.assertRaisesRegex(kv.ParkingError, "RAM evidence"):
            kv.validate_ram_evidence(evidence, self.contract)
        with self.assertRaisesRegex(kv.ParkingError, "disk evidence"):
            kv.validate_disk_evidence(ram_evidence(self.contract), self.contract, tokens)

    def test_disk_restore_order_and_seed_immutability_rechecks_are_strict(self):
        evidence, tokens, tails = disk_evidence(self.contract)
        evidence["restore_order"][0] = "B"
        with self.assertRaisesRegex(kv.ParkingError, "restore order"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)
        evidence, tokens, tails = disk_evidence(self.contract)
        del evidence["operations"][3]["seed_sha256_before_restore"]
        with self.assertRaisesRegex(kv.ParkingError, "not rechecked before"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)
        evidence, tokens, tails = disk_evidence(self.contract)
        evidence["operations"][2]["checkpoint_tail_reread_sha256"] = "0" * 64
        with self.assertRaisesRegex(kv.ParkingError, "pre-restore checkpoint-tail"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)

    def test_disk_requires_complete_session_progress_and_result(self):
        evidence, tokens, tails = disk_evidence(self.contract)
        evidence["operations"][0]["protocol_lines"] = ["SESSION 50 100", "SAVED 300 100 2.0"]
        with self.assertRaisesRegex(kv.ParkingError, "complete file size"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)
        evidence, tokens, tails = disk_evidence(self.contract)
        evidence["operations"][0]["protocol_lines"].append("SERR storage 0 no")
        with self.assertRaises(kv.ParkingError):
            kv.validate_disk_evidence(evidence, self.contract, tokens)

    def test_session_protocol_parser_is_bounded_and_strict(self):
        self.assertEqual(kv.parse_session_line("SESSION 16 16")["kind"], "progress")
        self.assertEqual(kv.parse_session_line("SWAIT publish 3")["kind"], "wait")
        self.assertEqual(kv.parse_session_line("SAVED 30 4096 1.5")["action"], "save")
        for line in ("SESSION -1 20", "SWAIT transfer 1801", "SAVED 1 1 nan", "SAVED 1 1 1 extra", "FATAL transfer"):
            with self.subTest(line=line), self.assertRaises(kv.ParkingError): kv.parse_session_line(line, 1800)


class GuardTests(unittest.TestCase):
    def test_ram_commit_gate_requires_known_psapi_floor(self):
        good = {"physical_available_bytes": 13 * 1024**3, "commit_available_bytes": 5 * 1024**3,
                "source": "GlobalMemoryStatusEx physical + PSAPI GetPerformanceInfo system commit"}
        self.assertEqual(kv.resource_gate(good)["status"], "pass")
        self.assertEqual(kv.resource_gate({**good, "commit_available_bytes": True})["status"], "fail_closed")
        self.assertEqual(kv.resource_gate({**good, "source": "psutil fallback"})["status"], "fail_closed")
        self.assertEqual(kv.resource_gate(good, additional_physical_bytes=2 * 1024**3)["status"], "fail_closed")

    def test_disk_space_floor_is_typed_and_fail_closed(self):
        c, _, _ = kv.load_contract()
        floor = c["resource_budget"]["initial_disk_free_required_bytes"]
        self.assertEqual(kv.disk_space_gate(floor, c)["status"], "pass")
        self.assertEqual(kv.disk_space_gate(floor - 1, c)["status"], "fail_closed")
        self.assertEqual(kv.disk_space_gate(True, c)["status"], "fail_closed")

    def test_owner_tree_requires_full_ordered_argv_and_create_time(self):
        expected = owners()
        self.assertEqual([x["role"] for x in kv.validate_owner_snapshot(expected, expected)], ["engine", "python-child", "launcher"])
        changed = copy.deepcopy(expected); changed[1]["argv"].reverse()
        with self.assertRaisesRegex(kv.ParkingError, "argv"): kv.validate_owner_snapshot(expected, changed)
        changed = copy.deepcopy(expected); changed[2]["create_time_utc"] = "2026-10-09T00:00:01Z"
        with self.assertRaisesRegex(kv.ParkingError, "create_time_utc"): kv.validate_owner_snapshot(expected, changed)

    def test_exact_cleanup_is_child_first_and_never_stops_foreign_pid(self):
        expected = owners(); live = copy.deepcopy(expected); stopped = []
        def observe(): return copy.deepcopy(live)
        def stop(identity):
            stopped.append(identity["role"])
            live[:] = [row for row in live if row["pid"] != identity["pid"]]
            return "exit0"
        kv.cleanup_exact_owner_tree(expected, observe, stop)
        self.assertEqual(stopped, ["engine", "python-child", "launcher"])
        live = copy.deepcopy(expected) + [{**expected[0], "pid": 999, "role": "foreign"}]
        stopped.clear()
        with self.assertRaisesRegex(kv.ParkingError, "outside the exact owned tree"):
            kv.cleanup_exact_owner_tree(expected, lambda: live, stop)
        self.assertEqual(stopped, [])

    def test_checkpoint_gap_and_unknown_owner_samples_fail(self):
        good = [{"resource": {"status": "pass"}, "owner_tree": owners(), "observed_at_utc": "2026-10-09T00:00:00Z"},
                {"resource": {"status": "pass"}, "owner_tree": owners(), "observed_at_utc": "2026-10-09T00:00:01Z"}]
        self.assertEqual(kv.validate_checkpoint_history(good)["sample_count"], 2)
        bad = copy.deepcopy(good); bad[1]["observed_at_utc"] = "2026-10-09T00:00:04Z"
        with self.assertRaisesRegex(kv.ParkingError, "cadence"): kv.validate_checkpoint_history(bad)
        bad = copy.deepcopy(good); bad[0]["resource"]["status"] = "fail_closed"
        with self.assertRaisesRegex(kv.ParkingError, "checkpoint failed"): kv.validate_checkpoint_history(bad)


class VersionTwoEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.contract_sha, cls.configs = kv.load_contract(kv.V2_CONTRACT)
        cls.request_doc = make_request_manifest(cls.contract, cls.contract_sha)
        cls.request_doc["schema_version"] = 2
        cls.request_doc["stdio_protocol_binding"] = copy.deepcopy(cls.contract["stdio_protocol_binding"])
        cls.requests_sha = hashlib.sha256(kv.canonical_json_bytes(cls.request_doc)).hexdigest()
        cls.requests = kv.validate_request_manifest(cls.request_doc, cls.contract, cls.contract_sha)
        cls.references_doc = make_reference_manifest(cls.contract, cls.contract_sha, cls.request_doc, cls.requests_sha)
        cls.references = kv.validate_reference_manifest(cls.references_doc, cls.requests, cls.contract,
                                                        cls.contract_sha, cls.requests_sha)

    def test_v2_plan_remains_unauthorized_and_default_off(self):
        plan = kv.build_plan(self.contract, self.contract_sha, self.configs,
                             self.requests_sha, None, kv.V2_CONTRACT)
        self.assertEqual(plan["execution_contract_version"], 2)
        self.assertFalse(plan["execution_authorized"])
        self.assertIn("prompt_tokenizer_calibration_not_run", plan["blockers"])

    def test_v2_protocol_stop_binding_is_mandatory_and_exact(self):
        doc = copy.deepcopy(self.request_doc)
        doc["stdio_protocol_binding"]["stop_token_ids"] = [248044, 248046]
        with self.assertRaisesRegex(kv.ParkingError, "stop bindings"):
            kv.validate_request_manifest(doc, self.contract, self.contract_sha)
        doc = copy.deepcopy(self.request_doc); del doc["stdio_protocol_binding"]
        with self.assertRaisesRegex(kv.ParkingError, "stop bindings"):
            kv.validate_request_manifest(doc, self.contract, self.contract_sha)

    def test_v2_actual_ids_are_the_quality_gate_and_fnv_is_additional(self):
        actual = completion_bundle(self.references_doc)
        result = kv.validate_arm_completions(actual, self.requests, self.references, self.contract)
        self.assertEqual(result["status"], "pass")
        actual["B"][1]["emitted_token_ids"][0] += 1
        with self.assertRaisesRegex(kv.ParkingError, "actual emitted token IDs"):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)

    def test_optional_post_gen_fingerprint_is_typed_and_compared_when_present(self):
        actual = completion_bundle(self.references_doc)
        actual["A"][0]["state_fingerprint"] = state_fingerprint(31000)
        self.assertEqual(kv.validate_arm_completions(actual, self.requests, self.references,
                         self.contract)["post_gen_fingerprint_missing"], 8)
        refs = copy.deepcopy(self.references_doc)
        refs["conversations"]["A"][0]["state_fingerprint"] = state_fingerprint(31000)
        bound = kv.validate_reference_manifest(refs, self.requests, self.contract, self.contract_sha, self.requests_sha)
        actual["A"][0]["state_fingerprint"] = state_fingerprint(31001)
        with self.assertRaisesRegex(kv.ParkingError, "post-GEN metadata"):
            kv.validate_arm_completions(actual, self.requests, bound, self.contract)

    def test_invalid_fnv_types_and_invented_sha_tail_are_rejected(self):
        actual = completion_bundle(self.references_doc)
        actual["A"][0]["state_fingerprint"] = state_fingerprint(31000)
        actual["A"][0]["state_fingerprint"]["algorithm"] = "sha256"
        with self.assertRaisesRegex(kv.ParkingError, "typed post-GEN"):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)
        actual = completion_bundle(self.references_doc)
        actual["A"][0]["checkpoint_tail_sha256"] = "a" * 64
        with self.assertRaisesRegex(kv.ParkingError, "checkpoint-tail digest"):
            kv.validate_arm_completions(actual, self.requests, self.references, self.contract)

    def test_disk_requires_full_hash_and_byte_recheck_on_every_restore(self):
        evidence, tokens, _ = disk_evidence(self.contract)
        self.assertEqual(kv.validate_disk_evidence(evidence, self.contract, tokens)["status"], "pass")
        bad = copy.deepcopy(evidence); bad["operations"][2]["seed_sha256_before_restore"] = "0" * 64
        with self.assertRaisesRegex(kv.ParkingError, "SHA-256 and byte count"):
            kv.validate_disk_evidence(bad, self.contract, tokens)

    def test_disk_seed_readback_hash_is_mandatory(self):
        evidence, tokens, _ = disk_evidence(self.contract)
        evidence["seed_files"][0]["readback_sha256"] = "0" * 64
        with self.assertRaisesRegex(kv.ParkingError, "readback or immutability hash"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)
        evidence, tokens, _ = disk_evidence(self.contract)
        del evidence["seed_files"][0]["sha256"]
        with self.assertRaisesRegex(kv.ParkingError, "lowercase SHA-256"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)
        evidence, tokens, _ = disk_evidence(self.contract)
        bad = copy.deepcopy(evidence); del bad["operations"][5]["seed_bytes_before_restore"]
        with self.assertRaisesRegex(kv.ParkingError, "SHA-256 and byte count"):
            kv.validate_disk_evidence(bad, self.contract, tokens)

    def test_disk_save_restore_token_and_file_byte_counts_are_strict(self):
        evidence, tokens, _ = disk_evidence(self.contract)
        evidence["operations"][0]["protocol_lines"][-1] = "SAVED 1 100 1.0"
        with self.assertRaisesRegex(kv.ParkingError, "SAVED token count"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)
        evidence, tokens, _ = disk_evidence(self.contract)
        evidence["operations"][2]["protocol_lines"][-1] = "RESTORED 31000 99 1.0"
        with self.assertRaisesRegex(kv.ParkingError, "RESTORED token/byte count"):
            kv.validate_disk_evidence(evidence, self.contract, tokens)


def state_fingerprint(length):
    domains = {key: f"{i:016x}" for i, key in enumerate(
        ("gdn", "ple", "tail", "dead", "pooled", "pooled_full", "kv", "mtp", "stale"), 1)}
    metadata = {"L_tokens": length, "ple_prev": [3, 4]}
    return {"algorithm": "fnv1a64", "phase": "post_gen", "cryptographic": False,
            "domains": domains, "metadata": metadata}


def make_request_manifest(contract, contract_sha):
    conversations = {}
    for conv in ("A", "B"):
        turns = []
        for index, turn_id in enumerate(kv.TURN_IDS):
            prompt = f"{conv} exact synthetic fixture prompt {turn_id}"
            ids = list(range(contract["context"]["base_prompt_target_tokens"])) if turn_id == "warmup" else [index + 1, index + 2]
            turns.append({"turn_id": turn_id, "prompt_utf8": prompt,
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "prompt_token_ids": ids, "max_tokens": contract["context"]["max_generated_per_turn"],
                          "sampling": {"temperature": 0, "top_p": 1, "seed": 42}})
        conversations[conv] = {"prompt_cache": 3, "turns": turns}
    return {"schema_version": 1, "contract_sha256": contract_sha, "conversations": conversations}


def _fixture_completion(conv, turn_id, request, expected):
    row = {"request_id": request["request_id"], "prompt_sha256": request["prompt_sha256"],
            "prompt_token_ids": list(request["prompt_token_ids"]), "max_tokens": request["max_tokens"],
            "sampling": request["sampling"], "prompt_cache_entries": 3,
            "raw_output_utf8": json.dumps(expected, ensure_ascii=False),
            "emitted_token_ids": [ord(conv), kv.TURN_IDS.index(turn_id) + 1],
            "canonical_reencoded_token_ids": [999999], "finish_reason": "stop",
            "state_fingerprint": None}
    if request.get("stop_token_ids") is not None: row["stop_token_ids"] = list(request["stop_token_ids"])
    return row


def make_reference_manifest(contract, contract_sha, request_doc, request_sha):
    turns = {}
    for conv in ("A", "B"):
        turns[conv] = [_fixture_completion(conv, turn_id,
            {"request_id": f"{conv}-{turn_id}",
             "prompt_sha256": next(x for x in request_doc["conversations"][conv]["turns"] if x["turn_id"] == turn_id)["prompt_sha256"],
             "prompt_token_ids": next(x for x in request_doc["conversations"][conv]["turns"] if x["turn_id"] == turn_id)["prompt_token_ids"],
             "max_tokens": contract["context"]["max_generated_per_turn"],
             "sampling": {"temperature": 0, "top_p": 1, "seed": 42}, "prompt_cache": 3,
             "stop_token_ids": list(request_doc["stdio_protocol_binding"]["stop_token_ids"]) if contract.get("schema_version") == 2 else None},
            contract["canonical_outputs"][conv]) for turn_id in kv.TURN_IDS]
    result = {"schema_version": contract.get("schema_version", 1), "contract_sha256": contract_sha,
              "request_manifest_sha256": request_sha, "conversations": turns}
    if contract.get("schema_version") == 2: result["stdio_protocol_binding"] = copy.deepcopy(contract["stdio_protocol_binding"])
    return result


def completion_bundle(reference_doc):
    return {conv: copy.deepcopy(reference_doc["conversations"][conv]) for conv in ("A", "B")}


def ram_evidence(contract):
    events = []
    for conv in ("A", "B"):
        for _ in range(3):
            tokens = contract["context"]["base_prompt_target_tokens"]
            events.append({"tier": "ram-hot", "conversation": conv, "action": "park", "cache_bytes": 1024,
                           "snapshot_bytes": 512, "admitted": True, "fallback": False,
                           "log_line": f"strata serve: conversation cache: parked {tokens} tokens in 1.0 ms; parked=1 bytes=1024 evictions=0 snapshot_bytes=512 reused_kv_bytes=256"})
            events.append({"tier": "ram-hot", "conversation": conv, "action": "restore", "cache_bytes": 1024,
                           "reused_prefix_tokens": tokens, "admitted": True, "fallback": False,
                           "log_line": f"strata serve: conversation cache: restored {tokens} tokens (checkpoint) in 1.0 ms; parked=1 bytes=1024"})
    return {"tier": "ram-hot", "request_sequence": list(contract["phases"][1]["sequence"]),
            "fallback_or_skip": [], "events": events, "repark_retained_bytes": {"A": 1024, "B": 1024}}


def disk_evidence(contract):
    directory = Path(contract["snapshot_paths"]["directory"])
    seed_hashes = {"A": "a" * 64, "B": "b" * 64}
    tokens = {"A": 31000, "B": 31100}
    seeds = [{"conversation": conv, "path": str(directory / f"{conv}-seed.bin"), "bytes": 100,
              "sha256": seed_hashes[conv], "readback_sha256": seed_hashes[conv], "after_sha256": seed_hashes[conv],
              "destination_absent_before_save": True, "created_exclusive": True,
              "before_file_id": conv, "after_file_id": conv, "before_mtime_ns": 1, "after_mtime_ns": 1,
              "regular_single_link_no_reparse": True}
             for conv in ("A", "B")]
    ops = []
    sequence = [("save", "A"), ("save", "B")] + [("restore", c) for c in ("A", "B", "A", "B", "A", "B")]
    for action, conv in sequence:
        result_verb = "SAVED" if action == "save" else "RESTORED"
        row = {"tier": "disk-parked-cold", "action": action, "conversation": conv,
               "seed_sha256": seed_hashes[conv],
               "protocol_lines": ["SESSION 0 100", "SESSION 100 100", "SWAIT transfer 1",
                                  f"{result_verb} {tokens[conv]} 100 1.0"]}
        if action == "restore":
            row.update({"seed_sha256_before_restore": seed_hashes[conv], "seed_bytes_before_restore": 100,
                        "seed_sha256_after_restore": seed_hashes[conv], "seed_bytes_after_restore": 100})
        ops.append(row)
    return ({"tier": "disk-parked-cold", "seed_files": seeds, "save_order": ["A", "B"],
             "restore_order": ["A", "B", "A", "B", "A", "B"], "operations": ops}, tokens, {})


def owners():
    return [{"role": "launcher", "pid": 101, "create_time_utc": "2026-10-09T00:00:00Z",
             "executable": r"C:\Python\python.exe", "parent_pid": 1, "argv": [r"C:\Python\python.exe", "launcher.py"]},
            {"role": "python-child", "pid": 102, "create_time_utc": "2026-10-09T00:00:01Z",
             "executable": r"C:\Python\python.exe", "parent_pid": 101, "argv": [r"C:\Python\python.exe", "child.py", "--run"]},
            {"role": "engine", "pid": 103, "create_time_utc": "2026-10-09T00:00:02Z",
             "executable": r"E:\build\strata.exe", "parent_pid": 102, "argv": [r"E:\build\strata.exe", "--private-stdio"]}]


if __name__ == "__main__":
    unittest.main()
