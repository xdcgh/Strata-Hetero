"""Synthetic-only checks for the payload exporter and CPU comparison harness."""
from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import struct
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np
from tools import hetero_resources as resources
from tools import hetero_vision_cpu_compare as compare
from tools import hetero_vision_payload as payload
from tools.test_hetero_vision_qwen3vl import small_fixture


def fake_export_contract(spec, inventory, fixtures, metadata):
    root = payload.DATA_ROOT / "SYNTHETIC_METADATA_ONLY_NO_PAYLOAD"
    tensors = {}
    for entry in inventory["tensor_descriptors"]:
        header = spec.headers[entry["name"]]
        tensors[header.name] = {
            "path": str(root / (header.name + ".npy")), "shape": list(header.numpy_shape), "dtype": "float32",
            "storage_type": header.storage_type, "raw_offset": entry["data_offset_absolute"],
            "raw_bytes": entry["bytes"], "raw_sha256": "a" * 64, "decoded_sha256": "b" * 64,
            "file_sha256": "c" * 64, "file_bytes": 128 + 4 * int(np.prod(header.numpy_shape)),
        }
    inputs = []
    for fixture in fixtures["fixtures"]:
        shape = [1, 3, fixture["height"], fixture["width"]]
        inputs.append({**fixture, "input": {
            "path": str(root / (fixture["name"] + ".input.npy")), "shape": shape, "dtype": "float32",
            "file_bytes": 128 + 4 * int(np.prod(shape)), "decoded_sha256": "b" * 64, "file_sha256": "c" * 64,
        }})
    return {
        **metadata, "status": "exported", "asset_sha256_verified": True, "output_directory": str(root),
        "decoder_hashes": payload.DECODER_HASHES, "tensors": tensors, "inputs": inputs,
        "identity": {"schema_version": 1, "kind": "gguf_decoded", "asset_sha256": spec.asset_sha256,
            "source_revision": spec.source_revision, "layout": payload.port.ARRAY_LAYOUT, "decoded_dtype": "float32",
            "decoder": f"llama.cpp:{payload.port.SOURCE_REVISION}:gguf.quants.dequantize",
            "weights_sha256": {name: "b" * 64 for name in tensors}},
    }


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-pure-preparation-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_both_default_clis_read_only_json_without_compute_imports_or_asset_access(self):
        original_import, original_open = builtins.__import__, Path.open

        def deny_runtime(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "openvino", "gguf", "PIL"):
                raise AssertionError("default imported a payload/device runtime")
            return original_import(name, *args, **kwargs)

        def json_only(path, *args, **kwargs):
            self.assertEqual(path.suffix, ".json", f"default opened payload: {path}")
            return original_open(path, *args, **kwargs)

        for module in (payload, compare):
            stdout = io.StringIO()
            with mock.patch("builtins.__import__", side_effect=deny_runtime), \
                    mock.patch.object(Path, "open", json_only), mock.patch("sys.stdout", stdout):
                self.assertEqual(module.main([]), 0)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["status"], "metadata_only")
            for flag in ("asset_opened", "payload_read", "files_written", "core_created", "compiled", "inference_run"):
                self.assertFalse(receipt[flag])

    def test_pinned_f32_bf16_decoder_with_small_known_bits_and_reverse_layout(self):
        decoder, _ = payload.pinned_decoder()  # Source import only; no GGUFReader instance.
        expected = np.array([[1, -2, 0], [0.5, 1.25, -0.5]], dtype=np.float32)
        for storage, raw in (
            ("F32", expected.tobytes()),
            ("BF16", (expected.view(np.uint32) >> 16).astype("<u2").tobytes()),
        ):
            header = payload.port.TensorHeader("SYNTHETIC", (3, 2), storage)
            decoded = payload.decode_tensor(raw, header, decoder)
            np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(type(decoded), np.ndarray)
            self.assertTrue(decoded.flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, "short raw"):
            payload.decode_tensor(b"\0", header, decoder)

    def test_owned_npy_readback_and_hash_corruption_gate(self):
        expected = np.array([[1, -2], [3.5, -4]], dtype=np.float32)
        artifact = payload.write_array(self.root / "fixture.npy", expected)
        np.testing.assert_array_equal(compare.load_array(artifact), expected)
        with self.assertRaises(FileExistsError):
            payload.write_array(self.root / "fixture.npy", expected)
        (self.root / "fixture.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "size/hash changed"):
            compare.load_array(artifact)

    def test_sve_parser_checks_hash_header_finite_and_exact_body(self):
        expected = np.array([[1, 2], [3, -4]], dtype=np.float32)
        raw = struct.pack("<5i", 0x31455653, 2, 2, 1, 2) + expected.tobytes()
        path = self.root / "synthetic.sve"
        path.write_bytes(raw)
        item = {"path": str(path), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "n_tokens": 2, "nx": 2, "ny": 1, "n_embd": 2}
        np.testing.assert_array_equal(compare.load_reference(item), expected)
        item["nx"] = 1
        with self.assertRaisesRegex(ValueError, "header differs"):
            compare.load_reference(item)
        item["nx"] = 2
        bad = struct.pack("<5i", 0x31455653, 2, 2, 1, 2) + np.full((2, 2), np.nan, np.float32).tobytes()
        path.write_bytes(bad)
        item["sha256"] = hashlib.sha256(bad).hexdigest()
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            compare.load_reference(item)

    def test_export_contract_validates_all_hash_layout_range_and_path_bindings(self):
        spec, inventory, fixtures, metadata = payload.inspect()
        exported = fake_export_contract(spec, inventory, fixtures, metadata)
        path = self.root / "SYNTHETIC-export.json"
        path.write_text(json.dumps(exported), encoding="utf-8")
        compare.inspect(export_path=path)
        for mutation in ("offset", "layout", "decoder", "path", "hash"):
            wrong = copy.deepcopy(exported)
            tensor = wrong["tensors"]["mm.0.bias"]
            if mutation == "offset":
                tensor["raw_offset"] += 32
            elif mutation == "layout":
                wrong["identity"]["layout"] = "unknown"
            elif mutation == "decoder":
                wrong["decoder_hashes"]["quants.py"] = "d" * 64
            elif mutation == "path":
                tensor["path"] = str(Path(wrong["output_directory"]).parent / "outside.npy")
            else:
                tensor["decoded_sha256"] = "a" * 64
            path.write_text(json.dumps(wrong), encoding="utf-8")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare.inspect(export_path=path)

    def test_native_gpu_or_flash_attention_receipts_are_refused(self):
        original = payload.read_json
        for mutation in ("gpu", "flash"):
            def altered(path):
                document, digest = original(path)
                if path.name == "vision-oracle-receipt.json":
                    if mutation == "gpu":
                        document["argv"].append("--gpu")
                    else:
                        document["argv"][-1] = "on"
                return document, digest
            with self.subTest(mutation=mutation), mock.patch.object(payload, "read_json", side_effect=altered):
                with self.assertRaisesRegex(ValueError, "completed CPU"):
                    compare.inspect()

    def test_new_alias_contract_validates_without_payload_or_runtime(self):
        _, _, _, _, metadata = compare.inspect(taps=("ln1.0", "ln1.1", "ln1.26", "qkv.0"))
        self.assertEqual(metadata["output_taps"], ["ln1.0", "ln1.1", "ln1.26", "qkv.0"])
        self.assertFalse(metadata["core_created"] or metadata["exported_arrays_read"])
        with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
            compare.inspect(taps=("ln1.2",))

    def test_failed_memory_gate_preserves_receipt_before_asset_or_core_access(self):
        spec, inventory, fixtures, metadata = payload.inspect()
        _, _, refs, _, comparison_metadata = compare.inspect()
        memory = {"physical_available_bytes": 11 * 1024**3, "commit_available_bytes": 8 * 1024**3, "source": "SYNTHETIC"}
        with mock.patch.object(payload, "DATA_ROOT", self.root), \
                mock.patch.object(resources, "memory_snapshot", return_value=(memory, [])), \
                mock.patch.object(payload, "pinned_decoder", side_effect=AssertionError("decoder started after failed gate")), \
                mock.patch.dict(sys.modules, {"openvino": None}):
            export = payload.prepare(spec, inventory, fixtures, metadata, self.root / "failed-export")
            compared = compare.run(spec, fixtures, refs, {}, comparison_metadata, self.root / "failed-compare")
        self.assertEqual(export["status"], "failed")
        self.assertFalse(export["asset_opened"])
        self.assertFalse(compared["core_created"])
        for folder, filename in (("failed-export", "export-receipt.json"), ("failed-compare", "comparison-receipt.json")):
            self.assertTrue((self.root / folder / filename).is_file())
            self.assertTrue((self.root / folder / "progress.jsonl").is_file())

    def test_header_mapping_closes_only_after_views_release_including_metadata_failure(self):
        for fail_metadata in (False, True):
            refs, closed = [], []
            class Mapping:
                def close(self):
                    self_test.assertTrue(all(ref() is None for ref in refs), "mapping closed with borrowed views alive")
                    closed.append(True)
            class Field:
                def contents(self):
                    if fail_metadata:
                        raise ValueError("synthetic metadata failure")
                    return [False]
            class Tensor:
                def __init__(self):
                    self.name, self.shape = "SYNTHETIC", np.array([3, 2], dtype=np.uint64)
                    self.tensor_type = types.SimpleNamespace(name="F32")
                    self.data_offset, self.n_bytes = 32, 24
            class Reader:
                def __init__(self, _asset, mode):
                    self.byte_order, self.data = "I", types.SimpleNamespace(_mmap=Mapping())
                    self.tensors, self.fields = [Tensor()], {"flag": Field()}
                    refs.extend(weakref.ref(obj) for obj in (self, self.tensors[0], self.tensors[0].shape, self.fields["flag"]))
            self_test = self
            with self.subTest(fail_metadata=fail_metadata):
                if fail_metadata:
                    with self.assertRaisesRegex(ValueError, "synthetic metadata failure"):
                        payload.fresh_headers(Reader, "SYNTHETIC_NO_FILE", {"SYNTHETIC"}, {"flag": {"value": [False]}})
                else:
                    fresh = payload.fresh_headers(Reader, "SYNTHETIC_NO_FILE", {"SYNTHETIC"}, {"flag": {"value": [False]}})
                    self.assertEqual(fresh, {"SYNTHETIC": ([3, 2], "F32", 32, 24)})
                self.assertEqual(closed, [True])

    def test_negative_quality_comparisons_keep_all_arms_with_unchanged_tolerances(self):
        spec, arrays, identity = small_fixture()
        fixtures = {"fixtures": [
            {"name": name, "height": 4, "width": width} for name, width in (("one", 4), ("two", 4), ("three", 4), ("wide", 8))
        ]}
        exported = {"identity": identity, "tensors": {name: {"token": name} for name in arrays},
                    "inputs": [{"name": f["name"], "input": {"token": f["name"]}} for f in fixtures["fixtures"]]}
        inputs = {f["name"]: np.zeros((1, 3, 4, f["width"]), np.float32) for f in fixtures["fixtures"]}
        refs = {f["name"]: {"name": f["name"], "shape": [f["width"] // 4, 5]} for f in fixtures["fixtures"]}
        props = types.SimpleNamespace(device=types.SimpleNamespace(full_name="FULL_DEVICE_NAME"),
            supported_properties="SUPPORTED_PROPERTIES", execution_devices="EXECUTION_DEVICES",
            hint=types.SimpleNamespace(inference_precision="INFERENCE_PRECISION_HINT", execution_mode="EXECUTION_MODE_HINT",
                                       ExecutionMode=types.SimpleNamespace(ACCURACY="ACCURACY")))
        calls = []

        class Compiled:
            def __init__(self, model): self.model = model
            def get_property(self, key):
                return {props.execution_devices: ["CPU"], props.hint.inference_precision: "f32", props.hint.execution_mode: "ACCURACY"}[key]
            def input(self, index): return "X"
            def create_infer_request(self):
                return types.SimpleNamespace(infer=lambda _inputs: calls.append(self.model),
                    get_output_tensor=lambda index: types.SimpleNamespace(data=np.ones(self.model["shape"], np.float32)))

        class Core:
            def get_property(self, device, key):
                self.assert_cpu(device)
                return "SYNTHETIC CPU" if key == props.device.full_name else [props.hint.inference_precision, props.hint.execution_mode]
            @staticmethod
            def assert_cpu(device): assert device == "CPU"
            def compile_model(self, model, device, settings):
                self.assert_cpu(device)
                assert settings == {props.hint.inference_precision: "f32", props.hint.execution_mode: "ACCURACY"}
                return Compiled(model)

        fake_ov = types.ModuleType("openvino")
        fake_ov.Core, fake_ov.properties, fake_ov.Type, fake_ov.__version__ = Core, props, types.SimpleNamespace(f32="f32"), "SYNTHETIC_HOST_FAKE"
        def fake_build(_spec, _arrays, _identity, **kwargs):
            shape = [kwargs["image_width"] // 4, 5]
            return {"shape": shape, "policy": kwargs["arithmetic_policy"]}, {"status": "SYNTHETIC_HOST_FAKE"}
        metadata = {"core_created": False, "compiled": False, "inference_run": False}
        memory = {"physical_available_bytes": 16 * 1024**3, "commit_available_bytes": 8 * 1024**3, "source": "SYNTHETIC"}
        with mock.patch.object(payload, "DATA_ROOT", self.root), \
                mock.patch.object(resources, "memory_snapshot", return_value=(memory, [])), \
                mock.patch.dict(sys.modules, {"openvino": fake_ov}), \
                mock.patch.object(compare, "load_array", side_effect=lambda item: arrays.get(item["token"], inputs.get(item["token"]))), \
                mock.patch.object(compare, "load_reference", side_effect=lambda item: np.zeros(item["shape"], np.float32)), \
                mock.patch.object(payload.port, "build_openvino_model", side_effect=fake_build):
            result = compare.run(spec, fixtures, refs, exported, metadata, self.root / "negative-host-fake")
        self.assertEqual(result["status"], "compared")
        self.assertEqual(len(result["arms"]), 4)
        self.assertEqual(len(calls), 32)
        self.assertFalse(result["all_quality_pass"])
        self.assertEqual(result["policy_quality_pass"], {"graph_f32": False, "cpu_vec_dot_rounding": False})
        self.assertEqual(result["quality_tolerances"], {"max_abs": 1e-2, "relative_rmse": 1e-3, "row_norm_relative": 1e-3})
        groups = [image for arm in result["arms"] for image in arm["images"]]
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(image["formal"]) == 3 and [f["iteration"] for f in image["formal"]] == [1, 2, 3] for image in groups))
        self.assertTrue(all("quality" not in image["warmup"] for image in groups))
        self.assertTrue(all(image["repeat_identical"] and len(image["repeat_hashes"]) == 3 for image in groups))
        self.assertTrue(all(not formal["quality"]["pass"] for image in groups for formal in image["formal"]))
        outputs = list((self.root / "negative-host-fake").glob("*.npy"))
        self.assertEqual(len(outputs), 24)
        self.assertTrue(all("-formal-" in path.name and "warmup" not in path.name for path in outputs))


if __name__ == "__main__":
    unittest.main()
