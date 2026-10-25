"""Host-only SXPU child for the compiled eligible fixture, never a model worker."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hetero_xpu_transport as transport
from tools.hetero_live_host_lifecycle import process_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descendant", action="store_true")
    parser.add_argument("--scenario", choices=("success", "reject", "partial", "short-output", "nan", "timeout"))
    parser.add_argument("--pids", type=Path)
    args = parser.parse_args(argv)
    if args.descendant:
        time.sleep(300)
        return 0
    if os.name != "nt" or not args.scenario or args.pids is None:
        parser.error("Windows scenario and exclusive PID receipt required")
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--descendant"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW)
    # The descendant deliberately outlives this bridge. The production C++ Job
    # Object, not this mock, must terminate it on failure/timeout/parent exit.
    with args.pids.open("x", encoding="utf-8") as stream:
        json.dump({"mock": process_identity(os.getpid()), "descendant": process_identity(child.pid), "scenario": args.scenario,
                   "synthetic_payload": True, "model_loaded": False, "device_accessed": False}, stream)
        stream.flush(); os.fsync(stream.fileno())
    frame = transport.read_frame(sys.stdin.buffer)
    if frame is None or frame.kind != 10:
        return 2
    if len(frame.payload) != 20 + 3072000 + 2*2560*4 or struct.unpack_from("<IIIII",frame.payload) != (28,288,2,2560,3072000):
        return 3
    if args.scenario == "timeout":
        time.sleep(300)
        return 4
    raw = frame.payload[20+3072000:]
    output = b"".join(struct.pack("<f", value[0]*2) for value in struct.iter_unpack("<f",raw))
    if args.scenario == "reject":
        body = struct.pack("<IQ",1,0)+b'{"message":"explicit mock rejection"}'
    else:
        if args.scenario == "nan":
            output = output[:-4]+struct.pack("<f",float("nan"))
        if args.scenario == "short-output":
            output = output[:-4]
        body = struct.pack("<IQ",0,123)+output
    wire = transport.encode_response_frame(frame.kind,frame.request_id,body)
    if args.scenario == "partial":
        wire = wire[:-16] # Full header advertises bytes that never arrive.
    transport.write_all(sys.stdout.buffer,wire)
    if args.scenario == "success":
        # Remain alive until fixture shutdown, demonstrating success cleanup.
        while transport.read_frame(sys.stdin.buffer) is not None:
            return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
