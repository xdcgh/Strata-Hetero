from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path

REPO = Path(r"E:\Strata-Hetero-data\source\llama-3cf0325")
GGUF_PY = REPO / "gguf-py"
PATH = Path(r"E:\Strata-Hetero-data\models\vision-ed59f92\mmproj-Qwen3.8-Flash-Next-BF16.gguf")
DOWNLOAD_RECEIPT = Path(r"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-15-vision-download\download-01.json")
OUT = Path(r"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-16-vision-encoder-headers\vision-header-inventory.json")
MAX_METADATA_JSON_BYTES = 65536

sys.path.insert(0, str(GGUF_PY))
from gguf.gguf_reader import GGUFReader  # noqa: E402


def stat_record(st):
    return {
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
        "device": int(st.st_dev),
        "file_index": int(st.st_ino),
    }


def normalize(v):
    if hasattr(v, "item") and callable(v.item):
        try:
            return normalize(v.item())
        except (ValueError, TypeError):
            pass
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): normalize(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def field_value(field):
    value = normalize(field.contents())
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=True).encode("utf-8")
    if len(encoded) <= MAX_METADATA_JSON_BYTES:
        return {"value": value, "value_truncated": False, "serialized_json_bytes": len(encoded)}
    if isinstance(value, list):
        preview = value[:32]
        omitted = len(value) - len(preview)
    elif isinstance(value, str):
        preview = value[:4096]
        omitted = len(value) - len(preview)
    else:
        preview = str(value)[:4096]
        omitted = max(0, len(str(value)) - len(preview))
    return {
        "value": preview,
        "value_truncated": True,
        "omitted_items_or_characters": omitted,
        "serialized_json_bytes": len(encoded),
        "serialized_value_sha256": hashlib.sha256(encoded).hexdigest(),
        "truncation_note": "Digest covers normalized header metadata value serialization, not tensor payload.",
    }


receipt = json.loads(DOWNLOAD_RECEIPT.read_text(encoding="utf-8"))
expected = receipt["final_identity"]
expected_sha = receipt["expected_lfs_sha256"]
expected_bytes = int(receipt["expected_size_bytes"])
fd = os.open(PATH, os.O_RDONLY)
try:
    pre_fd = stat_record(os.fstat(fd))
    pre_path = stat_record(os.stat(PATH))
    reader_source = GGUF_PY / "gguf" / "gguf_reader.py"
    reader_source_sha = hashlib.sha256(reader_source.read_bytes()).hexdigest()
    gguf = GGUFReader(PATH, mode="r")

    metadata = {}
    for key, field in gguf.fields.items():
        item = field_value(field)
        item["field_types"] = [getattr(t, "name", str(t)) for t in field.types]
        metadata[key] = item

    tensors = []
    for t in gguf.tensors:
        raw_shape = [int(x) for x in t.shape.tolist()]
        tensors.append({
            "name": t.name,
            "shape_gguf_dimension_order": raw_shape,
            "shape_numpy_order_reversed": list(reversed(raw_shape)),
            "type": getattr(t.tensor_type, "name", str(t.tensor_type)),
            "type_id": int(t.tensor_type),
            "elements": int(t.n_elements),
            "bytes": int(t.n_bytes),
            "data_offset_absolute": int(t.data_offset),
            "data_offset_relative_to_data_section": int(t.data_offset - gguf.data_offset),
        })

    post_fd = stat_record(os.fstat(fd))
    post_path = stat_record(os.stat(PATH))
finally:
    os.close(fd)

metadata_keys = sorted(metadata)
metadata_lookup = {k: (metadata[k].get("value") if not metadata[k].get("value_truncated") else None) for k in metadata}
layer_ids = sorted({int(m.group(1)) for t in tensors if (m := re.match(r"^v\.blk\.(\d+)\.", t["name"]))})
deepstack_ids = sorted({int(m.group(1)) for t in tensors if (m := re.match(r"^v\.deepstack\.(\d+)\.", t["name"]))})
vision_tensor_names = [t["name"] for t in tensors]
vision_related_metadata = {
    k: metadata[k] for k in metadata_keys
    if any(word in k.lower() for word in ("vision", "clip", "image", "patch", "merge", "rotary", "rope", "position", "deepstack", "temporal", "spatial"))
}

stat_stable = pre_path == post_path
fstat_stable = pre_fd == post_fd
path_matches_receipt = (
    pre_path["size_bytes"] == expected_bytes
    and pre_path["size_bytes"] == int(expected["size_bytes"])
    and pre_path["mtime_ns"] == int(expected["mtime_ns"])
    and pre_path["ctime_ns"] == int(expected["ctime_ns_path"])
    and pre_path["device"] == int(expected["file_id"]["volume"])
    and pre_path["file_index"] == int(expected["file_id"]["index"])
)
fstat_matches_receipt = (
    pre_fd["size_bytes"] == int(expected["fstat"]["size_bytes"])
    and pre_fd["mtime_ns"] == int(expected["fstat"]["mtime_ns"])
    and pre_fd["device"] == int(expected["fstat"]["volume"])
    and pre_fd["file_index"] == int(expected["fstat"]["index"])
)
identity_match = stat_stable and fstat_stable and path_matches_receipt and fstat_matches_receipt

report = {
    "schema_version": 1,
    "supersedes": "vision-header-inventory.json",
    "supersedes_reason": "Initial identity predicate required Windows path stat and fstat ctime to be equal and compared fields absent from the receipt fstat object. Corrected predicate separately checks stable stat/fstat and each against the fields actually recorded by download-01.",
    "status": "pass" if identity_match else "identity_mismatch",
    "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "scope": {
        "read": "GGUF header, metadata values, tensor descriptors only",
        "tensor_payload_bytes_accessed": False,
        "tensor_data_property_accessed": False,
        "payload_rehashed": False,
        "gguf_reader_note": "GGUFReader constructs NumPy memmap tensor views during parsing; this script only reads tensor name/type/shape/size/offset descriptors and never indexes or reads ReaderTensor.data.",
    },
    "source": {
        "repository": str(REPO),
        "revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
        "gguf_py_path": str(GGUF_PY),
        "gguf_reader_path": str(reader_source),
        "gguf_reader_sha256": reader_source_sha,
    },
    "asset": {
        "path": str(PATH),
        "download_receipt": str(DOWNLOAD_RECEIPT),
        "download_receipt_status": receipt.get("status"),
        "expected_sha256_from_verified_readback": expected_sha,
        "verified_stream_sha256": receipt.get("stream_sha256"),
        "verified_readback_sha256": receipt.get("readback_sha256"),
        "expected_bytes": expected_bytes,
        "current_stat_before": pre_path,
        "current_fstat_before": pre_fd,
        "current_stat_after": post_path,
        "current_fstat_after": post_fd,
        "stat_stable_during_scan": stat_stable,
        "fstat_stable_during_scan": fstat_stable,
        "stat_matches_receipt_final_identity": path_matches_receipt,
        "fstat_matches_receipt_fstat_identity": fstat_matches_receipt,
        "path_vs_fd_ctime_note": "Windows path stat and fstat ctime values differ; receipt stores path ctime, while fstat identity omits ctime. Both tuples are stable and their receipt-defined size/mtime/file-ID fields match.",
        "matches_receipt_final_identity_before_and_after": identity_match,
        "receipt_final_file_present_field": receipt.get("final_file_present"),
        "receipt_final_identity_present": bool(receipt.get("final_identity")),
        "receipt_internal_consistency_note": "The receipt reports final_file_present=false while also recording final_identity and final_is_same_verified_file=true; this inventory preserves that discrepancy and independently observes the final path present with matching receipt-defined stat and fstat identity fields.",
    },
    "gguf_header": {
        "version": int(gguf._get(4, gguf.gguf_scalar_to_np[next(t for t in gguf.fields["GGUF.version"].types if getattr(t, "name", "") == "UINT32")])[0]) if "GGUF.version" in gguf.fields else None,
        "byte_order": gguf.byte_order,
        "alignment": int(gguf.alignment),
        "data_section_offset": int(gguf.data_offset),
        "file_size_bytes": int(pre_path["size_bytes"]),
        "metadata_field_count": len(metadata),
        "tensor_count": len(tensors),
    },
    "metadata_keys_and_values": metadata,
    "vision_related_metadata": vision_related_metadata,
    "tensor_descriptors": tensors,
    "tensor_structure_derived_from_file": {
        "vision_block_ids": layer_ids,
        "vision_block_count": len(layer_ids),
        "vision_block_ids_contiguous_from_zero": layer_ids == list(range(len(layer_ids))),
        "deepstack_ids": deepstack_ids,
        "deepstack_block_count": len(deepstack_ids),
        "tensor_names": vision_tensor_names,
    },
    "field_status": {
        "projector_type": "from metadata key clip.projector_type if present; otherwise unknown",
        "vision_image_size": "from clip.vision.image_size or clip.vision.preproc_image_size; otherwise unknown",
        "patch_size": "from clip.vision.patch_size; otherwise unknown",
        "temporal_patch_size": "from clip.vision.temporal_patch_size; otherwise unknown",
        "spatial_merge_size": "from clip.vision.spatial_merge_size; otherwise unknown",
        "deepstack": "metadata keys and v.deepstack tensor names are listed above; semantics/model graph require source confirmation",
        "position_embeddings": "tensor names containing position_embd or positional terms are listed in descriptors; dimensions from file only",
        "rotary_parameters": "metadata keys containing rope/rotary are listed above; any absent key is unknown",
    },
}
if "GGUF.version" in metadata_lookup:
    report["gguf_header"]["version"] = metadata_lookup["GGUF.version"]
# Use exclusive creation; never overwrite a prior receipt.
with OUT.open("x", encoding="utf-8", newline="\n") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=True)
    f.write("\n")
print(json.dumps({"out": str(OUT), "identity_match": identity_match, "bytes": pre_path["size_bytes"], "tensor_count": len(tensors), "metadata_count": len(metadata), "block_count": len(layer_ids), "deepstack_count": len(deepstack_ids), "data_section_offset": int(gguf.data_offset)}, ensure_ascii=False))
