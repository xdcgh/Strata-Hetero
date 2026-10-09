#!/usr/bin/env python3
"""JSON-only validation by default; --prepare exports identified vision payloads."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import hetero_vision_qwen3vl as port
else:
    import hetero_vision_qwen3vl as port

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(r"E:\Strata-Hetero-data")
SOURCE_ROOT = DATA_ROOT / "source/llama-3cf0325"
DEFAULT_FIXTURES = ROOT / "bench/hetero/20261009-22-vision-oracle-cpu/fixture-manifest.json"
DECODER_HASHES = {
    "quants.py": "123b9d5741ede6ca8a321d7e08d43c8a0bd9fb53ec4f786323f5d9b9a214b4e8",
    "constants.py": "0bfc8c29cbcc3228fae0c64ae210cda9b14077abd72581961f7fda28c11b33e9",
    "gguf_reader.py": "495f86fe509eddef80b22e0d861de14d51ce5df01c5a584c63ddc302166eda8b",
}


class AdmissionError(RuntimeError):
    pass


def read_json(path):
    if path.suffix.lower() != ".json":
        raise ValueError("metadata paths must name JSON documents")
    with path.open("rb") as stream:
        raw = stream.read((2 << 20) + 1)
    if len(raw) > 2 << 20:
        raise ValueError("JSON document exceeds 2 MiB")
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def inspect(inventory_path=port.DEFAULT_INVENTORY, fixtures_path=DEFAULT_FIXTURES):
    inventory, inventory_sha = read_json(inventory_path)
    spec = port.spec_from_inventory(inventory)
    if inventory["gguf_header"]["byte_order"] != "I" or sys.byteorder != "little":
        raise ValueError("this exporter supports the selected little-endian GGUF only")
    end = inventory["gguf_header"]["data_section_offset"]
    for entry in sorted(inventory["tensor_descriptors"], key=lambda t: t["data_offset_absolute"]):
        size = math.prod(entry["shape_gguf_dimension_order"]) * (4 if entry["type"] == "F32" else 2)
        offset = entry["data_offset_absolute"]
        if type(offset) is not int or offset < end or size != entry["bytes"] or offset + size > port.ASSET_BYTES:
            raise ValueError("tensor range overlaps or exceeds the identified asset")
        end = offset + size
    fixtures, fixture_sha = read_json(fixtures_path)
    if fixtures.get("schema") != "strata-vision-fixtures-v1" or len(fixtures.get("fixtures", [])) != 4:
        raise ValueError("expected the four identified native-oracle fixtures")
    names = set()
    for item in fixtures["fixtures"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item["name"]) or item["name"] in names:
            raise ValueError("fixture names must be unique safe identifiers")
        names.add(item["name"])
        if (item["height"], item["width"]) not in ((96, 96), (96, 192)) or item["mode"] != "RGB" or item["format"] != "PNG":
            raise ValueError("only aligned RGB PNG fixtures at 96x96 or H96/W192 are supported")
        port._digest(item["sha256"], "fixture.sha256")
        port.image_shape_plan(spec, item["height"], item["width"])
    return spec, inventory, fixtures, {
        "schema_version": 1, "status": "metadata_only", "asset_sha256": spec.asset_sha256,
        "source_revision": spec.source_revision, "inventory_sha256": inventory_sha,
        "fixture_manifest_sha256": fixture_sha, "tensor_count": len(spec.tensors), "fixture_count": 4,
        "asset_opened": False, "payload_read": False, "files_written": False, "core_created": False,
        "compiled": False, "inference_run": False, "npu_accepted": False,
    }


def stable_stat(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def hash_file(path):
    before = path.stat()
    with path.open("rb") as stream:
        if stable_stat(os.fstat(stream.fileno())) != stable_stat(before):
            raise ValueError("opened file differs from checked identity")
        digest = hash_stream(stream)
        if stable_stat(os.fstat(stream.fileno())) != stable_stat(before):
            raise ValueError("file identity changed during hashing")
    if stable_stat(path.stat()) != stable_stat(before):
        raise ValueError("path identity changed during hashing")
    return digest


def hash_stream(stream, gate=None):
    stream.seek(0)
    digest = hashlib.sha256()
    for i, block in enumerate(iter(lambda: stream.read(8 << 20), b"")):
        digest.update(block)
        if gate is not None and i % 16 == 0:
            gate("asset-hash-chunk")
    return digest.hexdigest()


class Journal:
    def __init__(self, output):
        output = output.resolve(strict=False)
        if not output.is_relative_to(DATA_ROOT.resolve()) or output == DATA_ROOT.resolve():
            raise ValueError("new output directory must be within E:\\Strata-Hetero-data")
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.stage = output, "created"
        self.stream = (output / "progress.jsonl").open("x", encoding="utf-8")

    def event(self, stage, **fields):
        self.stage = stage
        self.stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def gate(self, stage):
        if __package__:
            from .hetero_resources import memory_snapshot
        else:
            from hetero_resources import memory_snapshot
        memory, errors = memory_snapshot()
        ram, commit = memory.get("physical_available_bytes"), memory.get("commit_available_bytes")
        passed = not errors and type(ram) is int and type(commit) is int and ram >= 12 * 1024**3 and commit >= 4 * 1024**3
        self.event(stage, resource_gate={"passed": passed, "memory": memory, "errors": errors})
        if not passed:
            raise AdmissionError("12 GiB RAM / 4 GiB commit gate failed or unknown")

    def finish(self, record, filename):
        record["last_stage"] = self.stage
        with (self.output / filename).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        self.stream.close()


def pinned_decoder():
    folder = SOURCE_ROOT / "gguf-py/gguf"
    for name, digest in DECODER_HASHES.items():
        if hash_file(folder / name) != digest:
            raise ValueError(f"pinned decoder source changed: {name}")
    sys.path.insert(0, str(folder.parent))
    decoder = importlib.import_module("gguf.quants")
    reader = importlib.import_module("gguf.gguf_reader")
    if Path(decoder.__file__).resolve() != (folder / "quants.py").resolve() or Path(reader.__file__).resolve() != (folder / "gguf_reader.py").resolve():
        raise ValueError("import resolved to a different GGUF decoder")
    return decoder, reader.GGUFReader


def decode_tensor(raw, header, decoder):
    import numpy as np
    expected = math.prod(header.gguf_shape) * {"F32": 4, "BF16": 2}[header.storage_type]
    if len(raw) != expected:
        raise ValueError("short raw tensor payload")
    decoded = decoder.dequantize(np.frombuffer(raw, np.uint8), decoder.GGMLQuantizationType[header.storage_type])
    array = np.array(decoded.reshape(header.numpy_shape), dtype=np.float32, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError("decoded tensor contains NaN or infinity")
    return array


def write_array(path, array):
    import numpy as np
    decoded_sha = port.decoded_array_sha256(array)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    reread = np.load(path, allow_pickle=False)
    if port.decoded_array_sha256(reread) != decoded_sha:
        raise ValueError("independent exported-array readback differs")
    return {"path": str(path), "shape": list(array.shape), "dtype": "float32",
            "decoded_sha256": decoded_sha, "file_sha256": hash_file(path), "file_bytes": path.stat().st_size}


def fresh_headers(reader_type, asset, expected_names, expected_metadata):
    """Copy header values, release borrowed views, then close the owned mapping."""
    reader = tensor = mapping = None
    error = None
    fresh = {}
    try:
        reader = reader_type(asset, mode="r")
        mapping = reader.data._mmap
        if reader.byte_order != "I":
            raise ValueError("fresh GGUF byte order differs")
        for tensor in reader.tensors:
            fresh[tensor.name] = (list(map(int, tensor.shape)), tensor.tensor_type.name,
                                  int(tensor.data_offset), int(tensor.n_bytes))
        if set(fresh) != set(expected_names):
            raise ValueError("fresh GGUF descriptor set differs")
        for key, entry in expected_metadata.items():
            if key not in reader.fields or reader.fields[key].contents() != entry["value"]:
                raise ValueError("fresh GGUF vision metadata differs: " + key)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"  # Do not retain tracebacks that own header views.
    finally:
        reader = tensor = None
        gc.collect()
        if mapping is not None:
            mapping.close()
    if error is not None:
        raise ValueError("fresh header read failed: " + error)
    return fresh


def prepare(spec, inventory, fixtures, metadata, output):
    journal = Journal(output)
    result = {**metadata, "status": "running", "files_written": True,
              "output_directory": str(journal.output), "tensors": {}, "inputs": []}
    try:
        journal.gate("before-asset-open")
        import numpy as np
        from PIL import Image, __version__ as pillow_version
        decoder, reader_type = pinned_decoder()
        result["decoder_hashes"] = DECODER_HASHES
        result["decoder"] = f"llama.cpp:{port.SOURCE_REVISION}:gguf.quants.dequantize"
        asset = Path(inventory["asset"]["path"])
        before = stable_stat(asset.stat())
        if before[2] != port.ASSET_BYTES:
            raise ValueError("selected asset size changed")
        with asset.open("rb") as stream:
            result.update(asset_opened=True, payload_read=True)
            if stable_stat(os.fstat(stream.fileno())) != before or hash_stream(stream, journal.gate) != port.ASSET_SHA256:
                raise ValueError("opened asset identity or whole SHA-256 differs")
            result.update(asset_opened=True, payload_read=True, asset_sha256_verified=True)
            fresh = fresh_headers(reader_type, asset, spec.headers, inventory["vision_related_metadata"])
            for entry in inventory["tensor_descriptors"]:
                journal.gate("before-tensor:" + entry["name"])
                if fresh[entry["name"]] != (
                        entry["shape_gguf_dimension_order"], entry["type"], entry["data_offset_absolute"], entry["bytes"]):
                    raise ValueError("fresh tensor descriptor differs from the JSON inventory")
                stream.seek(entry["data_offset_absolute"])
                raw = stream.read(entry["bytes"])
                array = decode_tensor(raw, spec.headers[entry["name"]], decoder)
                artifact = write_array(journal.output / (entry["name"] + ".npy"), array)
                result["tensors"][entry["name"]] = {**artifact, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_bytes": len(raw), "raw_offset": entry["data_offset_absolute"], "storage_type": entry["type"]}
                journal.event("tensor-exported", name=entry["name"], decoded_sha256=artifact["decoded_sha256"])
            if hash_stream(stream, journal.gate) != port.ASSET_SHA256 or stable_stat(os.fstat(stream.fileno())) != before:
                raise ValueError("source asset changed during export")
        if stable_stat(asset.stat()) != before:
            raise ValueError("source asset path identity changed during export")
        result["asset_stable_stat"] = list(before)
        for item in fixtures["fixtures"]:
            journal.gate("before-fixture:" + item["name"])
            image_path = Path(item["path"])
            if hash_file(image_path) != item["sha256"]:
                raise ValueError("fixture PNG hash changed")
            with Image.open(image_path) as image:
                if image.mode != "RGB" or image.format != "PNG" or image.size != (item["width"], item["height"]):
                    raise ValueError("fixture decoder shape/mode differs; no resize or conversion allowed")
                pixels = np.array(image, dtype=np.float32, copy=True) / np.float32(255)
            if hash_file(image_path) != item["sha256"]:
                raise ValueError("fixture PNG changed while decoded")
            pixels = (pixels - np.asarray(spec.image_mean, np.float32)) / np.asarray(spec.image_std, np.float32)
            image_input = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
            result["inputs"].append({**item, "input": write_array(journal.output / (item["name"] + ".input.npy"), image_input)})
        result["identity"] = {"schema_version": 1, "kind": "gguf_decoded", "asset_sha256": spec.asset_sha256,
            "source_revision": spec.source_revision, "decoder": result["decoder"], "layout": port.ARRAY_LAYOUT,
            "decoded_dtype": "float32", "weights_sha256": {name: t["decoded_sha256"] for name, t in result["tensors"].items()}}
        journal.gate("export-complete")
        result.update(status="exported", files_written=True, numpy_version=np.__version__, pillow_version=pillow_version,
                      preprocessing="Pillow identified aligned RGB PNG; no resize/pad; F32 /255 then stored mean/std then NCHW")
    except Exception as exc:
        result.update(status="failed", failure_stage=journal.stage, error=f"{type(exc).__name__}: {exc}")
    journal.finish(result, "export-receipt.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=port.DEFAULT_INVENTORY)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.prepare != bool(args.output_dir):
        parser.error("--prepare and --output-dir must be provided together")
    try:
        spec, inventory, fixtures, result = inspect(args.inventory, args.fixtures)
        if args.prepare:
            result = prepare(spec, inventory, fixtures, result, args.output_dir)
        print(json.dumps(result, indent=2))
        return 1 if result["status"] == "failed" else 0
    except Exception as exc:
        print(json.dumps({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
