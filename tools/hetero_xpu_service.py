#!/usr/bin/env python3
"""Standalone SXPU Arc service. Import/default CLI paths do not import OpenVINO or create a Core."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import hetero_xpu_transport as transport  # noqa: E402
from tools import hetero_xpu_worker as xpu_worker  # noqa: E402

SERVICE_INIT_SCHEMA = "strata-xpu-init-v1"
INIT_KIND = 1
INFER_KIND = 2
MAX_INIT_JSON_BYTES = 64 * 1024
MAX_ERROR_JSON_BYTES = 4096
RESULT_PREFIX = struct.Struct("<IQ")  # status:u32, worker_wall_ns:u64


class ServiceError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ServiceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_init(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_INIT_JSON_BYTES:
        raise ServiceError(f"INIT JSON exceeds {MAX_INIT_JSON_BYTES} bytes")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_no_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ServiceError(f"invalid INIT JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ServiceError("INIT must be a JSON object")
    expected = {"schema", "nonce", "rows", "identity_sha256"}
    if set(value) != expected:
        raise ServiceError(f"INIT keys must be exactly {sorted(expected)}")
    if value["schema"] != SERVICE_INIT_SCHEMA:
        raise ServiceError("unsupported INIT schema")
    nonce = value["nonce"]
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-fA-F]{32}", nonce) is None:
        raise ServiceError("INIT nonce must be exactly 32 hexadecimal characters")
    rows = value["rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 256:
        raise ServiceError("INIT rows must be an integer in 1..256")
    identity_sha = value["identity_sha256"]
    if not isinstance(identity_sha, str) or re.fullmatch(r"[0-9a-f]{64}", identity_sha) is None:
        raise ServiceError("INIT identity_sha256 must be lowercase 64-character SHA-256")
    return {"schema": SERVICE_INIT_SCHEMA, "nonce": nonce, "rows": rows,
            "identity_sha256": identity_sha}


def _error_json(message: str) -> bytes:
    # Bound the source string before serialization; slicing encoded JSON can
    # produce malformed UTF-8/JSON, especially around escaped Unicode.
    text = str(message)[:256]
    data = json.dumps({"message": text}, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if len(data) > MAX_ERROR_JSON_BYTES:
        data = b'{"message":"error"}'
    return data


def _result_frame(request_kind: int, request_id: int, status: int, worker_wall_ns: int, body: bytes) -> bytes:
    if status not in (0, 1):
        raise ServiceError("invalid result status")
    if isinstance(worker_wall_ns, bool) or not isinstance(worker_wall_ns, int) or not 0 <= worker_wall_ns <= 0xFFFFFFFFFFFFFFFF:
        raise ServiceError("worker_wall_ns is outside UInt64")
    return transport.encode_response_frame(request_kind, request_id, RESULT_PREFIX.pack(status, worker_wall_ns) + body)


class ArcWorkerService:
    """One fixed weights/identity/device service with one static-rows graph at a time."""

    def __init__(self, *, weights_path: Path, identity_path: Path, identity: dict[str, Any],
                 identity_sha256: str, header_info: dict[str, Any], core: Any, ov_api: Any,
                 worker_api=xpu_worker, device: str = "GPU.0", precision: str = "f32") -> None:
        if device != "GPU.0" or precision != "f32":
            raise ServiceError("service is restricted to exact GPU.0 with f32 precision")
        self.weights_path = Path(weights_path).resolve()
        self.identity_path = Path(identity_path).resolve()
        self.identity = identity
        self.identity_sha256 = identity_sha256
        self.header_info = header_info
        self.core = core
        self.ov = ov_api
        self.worker = worker_api
        self.device = device
        self.precision = precision
        self.hidden = int(header_info["hidden"])
        self._weights = None
        self._device_info: dict[str, Any] | None = None
        self._model = None
        self._compiled = None
        self._infer_request = None
        self._input_port = None
        self._output_port = None
        self._rows: int | None = None
        self._nonce: str | None = None
        self._ready_record: dict[str, Any] | None = None

    @property
    def active_rows(self) -> int | None:
        return self._rows

    @property
    def initialized(self) -> bool:
        return self._compiled is not None and self._infer_request is not None and self._nonce is not None

    def _verify_identity_file(self, expected_sha: str) -> None:
        if self.identity_path.is_symlink() or not self.identity_path.is_file():
            raise ServiceError("fixed identity path is missing or is a symlink")
        before = self.identity_path.stat()
        if before.st_size > MAX_INIT_JSON_BYTES:
            raise ServiceError("fixed identity JSON exceeds bounded metadata size")
        # Limit the read itself as well as the preflight stat to avoid an
        # unbounded allocation if the file grows between those operations.
        with self.identity_path.open("rb") as stream:
            raw = stream.read(MAX_INIT_JSON_BYTES + 1)
        if len(raw) > MAX_INIT_JSON_BYTES:
            raise ServiceError("fixed identity JSON grew beyond bounded metadata size")
        after = self.identity_path.stat()
        if (after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns or
                after.st_dev != before.st_dev or after.st_ino != before.st_ino):
            raise ServiceError("fixed identity file changed while being verified")
        if _sha256(raw) != self.identity_sha256 or expected_sha != self.identity_sha256:
            raise ServiceError("INIT identity_sha256 does not match the fixed identity file")
        try:
            current = self.worker._safe_identity(json.loads(raw.decode("utf-8")))
        except Exception as exc:
            raise ServiceError(f"fixed identity file is invalid: {exc}") from exc
        if current != self.identity:
            raise ServiceError("fixed identity file content changed after startup validation")

    def _verify_weights_stat(self) -> None:
        info = self.header_info
        stat = self.weights_path.stat()
        file_id = info["file_id"]
        if (stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"] or
                stat.st_dev != file_id["volume"] or stat.st_ino != file_id["index"]):
            raise ServiceError("fixed weights NPZ identity changed after header validation")

    def _probe_arc_device(self) -> dict[str, Any]:
        properties = self.ov.properties
        available = [str(name) for name in self.core.available_devices]
        if self.device not in available:
            raise ServiceError(f"required device {self.device} is not available: {available}")
        full_name = str(self.core.get_property(self.device, properties.device.full_name))
        folded = full_name.casefold()
        if "intel" not in folded or "arc" not in folded:
            raise ServiceError(f"GPU.0 full name is not Intel Arc: {full_name}")
        raw_supported = self.core.get_property(self.device, properties.supported_properties)
        supported = {str(name) for name in raw_supported}
        policy = self.worker.compile_property_policy(self.device, self.precision, supported)
        expected = {"INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"}
        if policy["submitted"] != expected:
            raise ServiceError(f"device did not accept the required f32/ACCURACY hints: {policy}")
        return {"full_name": full_name, "available_devices": available,
                "supported_properties": sorted(supported), "compile_property_policy": policy}

    def _release_graph(self) -> None:
        self._infer_request = None
        self._input_port = None
        self._output_port = None
        self._compiled = None
        self._model = None
        self._rows = None
        self._nonce = None
        self._ready_record = None

    def _compile_rows(self, rows: int) -> tuple[dict[str, Any], int]:
        if self._weights is None:
            self._verify_weights_stat()
            self._weights = self.worker.load_verified_weights(self.weights_path, self.identity, self.header_info)
        if self._device_info is None:
            self._device_info = self._probe_arc_device()

        hint = self.ov.properties.hint
        names = {"INFERENCE_PRECISION_HINT": hint.inference_precision,
                 "EXECUTION_MODE_HINT": hint.execution_mode}
        compile_properties = {}
        for name in self._device_info["compile_property_policy"]["submitted"]:
            compile_properties[names[name]] = (
                self.ov.Type.f32 if name == "INFERENCE_PRECISION_HINT" else hint.ExecutionMode.ACCURACY)

        start = time.perf_counter_ns()
        model = self.worker.make_ffn_model(self._weights, rows, "f32")
        compiled = self.core.compile_model(model, self.device, compile_properties)
        compile_ns = time.perf_counter_ns() - start

        raw_devices = compiled.get_property(self.ov.properties.execution_devices)
        if isinstance(raw_devices, str):
            actual_devices = [name.strip() for name in raw_devices.split(",") if name.strip()]
        else:
            actual_devices = [str(name) for name in raw_devices]
        if actual_devices != ["GPU.0"]:
            raise ServiceError(f"compiled graph execution devices must be exactly ['GPU.0'], got {actual_devices}")
        actual_precision = compiled.get_property(hint.inference_precision)
        precision_policy = self.worker.assess_precision_policy(actual_precision, self.ov.Type.f32)
        if not precision_policy["precision_policy_matches"]:
            raise ServiceError(f"compiled graph reported a non-matching precision: {actual_precision}")

        request = compiled.create_infer_request()
        self._model = model
        self._compiled = compiled
        self._infer_request = request
        self._input_port = compiled.input(0)
        self._output_port = compiled.output(0)
        self._rows = rows
        record = {
            "schema": "strata-xpu-ready-v1", "status": "ready", "rows": rows,
            "device": self.device, "device_full_name": self._device_info["full_name"],
            "available_devices": self._device_info["available_devices"],
            "execution_devices": actual_devices, "precision_requested": "f32",
            "reported_precision": str(actual_precision), "precision_policy": precision_policy,
            "compile_property_policy": self._device_info["compile_property_policy"],
            "compile_ns": compile_ns, "openvino_version": str(self.ov.__version__),
            "weights_loaded_and_hash_verified": True,
            "weights_sha256": self.identity["weights_sha256"],
        }
        return record, compile_ns

    def _handle_init(self, frame: transport.Frame) -> bytes:
        if len(frame.payload) > MAX_INIT_JSON_BYTES:
            raise ServiceError(f"INIT JSON exceeds {MAX_INIT_JSON_BYTES} bytes")
        request = _parse_init(frame.payload)
        self._verify_identity_file(request["identity_sha256"])
        input_bytes = request["rows"] * self.hidden * 4
        if input_bytes > transport.MAX_PAYLOAD_BYTES - RESULT_PREFIX.size:
            raise ServiceError("requested static rows/hidden shape exceeds the INFER frame limit")

        if self._compiled is not None and self._rows == request["rows"]:
            ready = dict(self._ready_record)
            ready["cached_graph_reused"] = True
            ready["compile_ns"] = 0
        else:
            # A rows change releases the old graph before compiling the replacement.
            if self._compiled is not None:
                self._release_graph()
            ready, _ = self._compile_rows(request["rows"])
            ready["cached_graph_reused"] = False
            self._ready_record = dict(ready)
        self._nonce = request["nonce"]
        ready["nonce"] = self._nonce
        ready["identity_sha256"] = self.identity_sha256
        return json.dumps(ready, ensure_ascii=True, separators=(",", ":")).encode("utf-8")

    def _handle_infer(self, frame: transport.Frame) -> tuple[bytes, int]:
        if not self.initialized or self._rows is None:
            raise ServiceError("INFER received before a successful INIT")
        expected = self._rows * self.hidden * 4
        if len(frame.payload) != expected:
            raise ServiceError(f"INFER payload size mismatch: expected {expected}, got {len(frame.payload)}")
        view = self.worker.np.frombuffer(frame.payload, dtype="<f4").reshape(self._rows, self.hidden)
        if not self.worker.np.isfinite(view).all():
            raise ServiceError("INFER input contains NaN or infinity")

        start = time.perf_counter_ns()
        owned_input = self.worker.np.array(view, dtype=self.worker.np.float32, copy=True, order="C")
        if not owned_input.flags.c_contiguous or not owned_input.flags.owndata:
            raise ServiceError("INFER input copy is not owned contiguous F32")
        self._infer_request.infer({self._input_port: owned_input})
        output_view = self._infer_request.get_output_tensor(0).data
        owned_output = self.worker.np.array(output_view, dtype=self.worker.np.float32, copy=True, order="C")
        output_bytes = owned_output.astype("<f4", copy=False).tobytes(order="C")
        worker_wall_ns = time.perf_counter_ns() - start

        if owned_output.shape != (self._rows, self.hidden):
            raise ServiceError(f"INFER output shape mismatch: {owned_output.shape}")
        if not self.worker.np.isfinite(owned_output).all():
            raise ServiceError("INFER output contains NaN or infinity")
        if len(output_bytes) != expected or len(output_bytes) > transport.MAX_PAYLOAD_BYTES - RESULT_PREFIX.size:
            raise ServiceError("INFER output exceeds the framed response limit")
        return output_bytes, worker_wall_ns

    def handle_frame(self, frame: transport.Frame) -> bytes:
        start = time.perf_counter_ns()
        if frame.magic != transport.MAGIC or frame.version != transport.VERSION:
            raise ServiceError("invalid frame magic/version")
        if frame.kind >= transport.RESPONSE_BIT:
            raise ServiceError("request kind must not have the response bit set")
        try:
            if frame.kind == INIT_KIND:
                body = self._handle_init(frame)
                status = 0
                worker_ns = time.perf_counter_ns() - start
            elif frame.kind == INFER_KIND:
                body, worker_ns = self._handle_infer(frame)
                status = 0
            else:
                raise ServiceError(f"unknown operation kind: {frame.kind}")
        except Exception as exc:
            if frame.kind == INIT_KIND:
                # An unsuccessful reconfiguration must never leave the old
                # nonce usable for INFER.
                self._nonce = None
            status = 1
            worker_ns = time.perf_counter_ns() - start
            body = _error_json(f"{type(exc).__name__}: {exc}")
        return _result_frame(frame.kind, frame.request_id, status, worker_ns, body)


def serve_stdio(service: ArcWorkerService, protocol_in: BinaryIO, protocol_out: BinaryIO) -> int:
    """Serial request loop. Malformed/unrecoverable streams are logged and closed."""
    while True:
        try:
            frame = transport.read_frame(protocol_in)
        except transport.FrameTooLarge as exc:
            print(f"hetero_xpu_service: {exc}", file=sys.stderr, flush=True)
            if 0 <= exc.kind < transport.RESPONSE_BIT:
                response = _result_frame(exc.kind, exc.request_id, 1, 0, _error_json(str(exc)))
                try:
                    transport.write_all(protocol_out, response)
                except Exception as write_exc:
                    print(f"hetero_xpu_service: cannot write frame-limit error: {write_exc}", file=sys.stderr, flush=True)
            return 2  # unread oversized payload prevents safe stream resynchronization
        except transport.ProtocolError as exc:
            print(f"hetero_xpu_service: protocol error: {exc}", file=sys.stderr, flush=True)
            return 2
        if frame is None:
            return 0
        if frame.kind >= transport.RESPONSE_BIT:
            print(f"hetero_xpu_service: request kind has response bit set: {frame.kind}", file=sys.stderr, flush=True)
            return 2
        try:
            transport.write_all(protocol_out, service.handle_frame(frame))
        except Exception as exc:
            print(f"hetero_xpu_service: request failed outside frame handler: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return 2


def _load_metadata(weights_path: Path, identity_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ServiceError("identity path is missing or is a symlink")
    if identity_path.stat().st_size > MAX_INIT_JSON_BYTES:
        raise ServiceError("identity JSON exceeds bounded metadata size")
    before_stat = identity_path.stat()
    raw = identity_path.read_bytes()
    try:
        identity = xpu_worker._safe_identity(json.loads(raw.decode("utf-8")))
    except Exception as exc:
        raise ServiceError(f"invalid identity JSON: {exc}") from exc
    header = xpu_worker.validate_npz_metadata(weights_path, identity)
    after_stat = identity_path.stat()
    after_raw = identity_path.read_bytes()
    if raw != after_raw or (before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_dev, before_stat.st_ino) != (
        after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_dev, after_stat.st_ino):
        raise ServiceError("identity file changed during header validation")
    return identity, header, _sha256(raw)


def _create_openvino_runtime():
    """This lazy import is reachable only after explicit --serve and stdout redirection."""
    ov = importlib.import_module("openvino")
    return ov, ov.Core()


def _redirect_stdout_keep_protocol() -> BinaryIO:
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(1), "wb", buffering=0)
    try:
        os.dup2(2, 1)
    except BaseException:
        protocol.close()
        raise
    return protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="default: inspect identity and NPZ headers only")
    mode.add_argument("--serve", action="store_true", help="explicitly start the Arc GPU.0 child service")
    parser.add_argument("--weights", type=Path, required=True, help="fixed NPZ path with gate/up/down F32 arrays")
    parser.add_argument("--identity", type=Path, required=True, help="fixed identity JSON path")
    parser.add_argument("--device", choices=("GPU.0",), help="serve mode requires exact GPU.0")
    parser.add_argument("--precision", choices=("f32",), help="serve mode requires f32")
    return parser


def main(argv: list[str] | None = None, *, runtime_factory: Callable[[], tuple[Any, Any]] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    protocol_out = None
    if "--serve" in raw_argv:
        try:
            protocol_out = _redirect_stdout_keep_protocol()
        except Exception as exc:
            print(f"hetero_xpu_service: protocol stdout setup failed: {exc}", file=sys.stderr, flush=True)
            return 2
    try:
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        if args.serve:
            if args.device != "GPU.0" or args.precision != "f32":
                parser.error("--serve requires explicit --device GPU.0 --precision f32")
            if not args.weights.is_absolute() or not args.identity.is_absolute():
                parser.error("--serve requires absolute --weights and --identity paths")
        identity, header, identity_sha = _load_metadata(args.weights, args.identity)
        if not args.serve:
            print(json.dumps({
                "status": "validated_only", "identity_sha256": identity_sha,
                "weights_header_info": header, "openvino_imported": False,
                "core_created": False, "device_queried": False, "weights_loaded": False,
                "file_written": False, "worker_started": False,
            }, indent=2, ensure_ascii=True))
            return 0
        if protocol_out is None:
            raise ServiceError("serve mode has no preserved binary stdout descriptor")
        factory = _create_openvino_runtime if runtime_factory is None else runtime_factory
        ov, core = factory()
        service = ArcWorkerService(weights_path=args.weights, identity_path=args.identity,
                                   identity=identity, identity_sha256=identity_sha,
                                   header_info=header, core=core, ov_api=ov)
        return serve_stdio(service, sys.stdin.buffer, protocol_out)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"hetero_xpu_service: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if protocol_out is not None:
            try:
                protocol_out.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
