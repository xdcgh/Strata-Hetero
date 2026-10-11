"""Authorized private-stdio KV parking adapter. Import has no model/device/API side effects."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import stat
import subprocess
import threading
import time
from typing import Any, Callable

from tools import hetero_kv_parking as kv

API_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
RUN42 = ROOT / "bench/hetero/20261009-42-native-q4k-q5_1-ipc/executed-client.py"
BIN_SHA = "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb"
SOURCE_SHA = "ffeb748d190ddeedff23587ce848596d4db0c752"
LINE_MAX = 1 << 20
LOG_MAX = 128 << 20
STATE_HASH_DOMAINS = ("gdn", "ple", "tail", "pooled", "kv", "mtp", "stale", "dead", "pooled_full")
STATE_SOURCE = ("L", *STATE_HASH_DOMAINS, "ple_prev")
STATE_RE = re.compile(r"(?:^| )STATE_HASH (?P<body>.*)$")


class StdioAdapterError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _helpers():
    if not RUN42.is_file(): raise StdioAdapterError(f"pinned run42 owner monitor missing: {RUN42}")
    spec = importlib.util.spec_from_file_location("hetero_kv_run42", RUN42)
    if spec is None or spec.loader is None: raise StdioAdapterError("cannot load run42 owner monitor")
    import sys
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


def _tokenizer(path: Path, binding: dict[str, Any]):
    # Delayed until execute_plan; only pack tokenizer files are read, never model weights.
    source = Path(binding["tokenizer_source_path"])
    if _sha(source.read_bytes()) != binding.get("tokenizer_source_sha256"):
        raise StdioAdapterError("pinned tokenizer implementation changed")
    spec = importlib.util.spec_from_file_location("kv_pinned_strata_tokenizer", source)
    if spec is None or spec.loader is None: raise StdioAdapterError("cannot load pinned tokenizer implementation")
    import sys
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    Tokenizer = module.Tokenizer
    files = {"tokenizer_config_sha256": "tokenizer.json", "vocab_sha256": "vocab.json",
             "token_type_sha256": "token_type.json", "merges_sha256": "merges.txt"}
    for field, name in files.items():
        if _sha((path / name).read_bytes()) != binding.get(field):
            raise StdioAdapterError(f"pinned tokenizer file changed: {name}")
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    words: list[str | None] = [None] * len(vocab)
    for word, idx in vocab.items():
        if type(idx) is not int or idx < 0 or idx >= len(words) or words[idx] is not None:
            raise StdioAdapterError("tokenizer vocabulary IDs are not dense and unique")
        words[idx] = word
    if any(x is None for x in words): raise StdioAdapterError("tokenizer vocabulary has an ID gap")
    merges = (path / "merges.txt").read_text(encoding="utf-8").splitlines()
    types = json.loads((path / "token_type.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer(words, merges, types)  # type: ignore[arg-type]
    stops = tokenizer.encode("<|im_end|>", parse_special=True) + tokenizer.encode("<|endoftext|>", parse_special=True)
    if stops != binding.get("stop_token_ids") or stops != [248046, 248044]:
        raise StdioAdapterError("tokenizer-derived stop IDs differ from the pinned bridge pair")
    bridge = Path(binding["bridge_path"])
    if _sha(bridge.read_bytes()) != binding.get("bridge_sha256"):
        raise StdioAdapterError("pinned server bridge changed")
    return tokenizer


def _owner(run42, proc) -> dict[str, Any]:
    row = run42._process_record(proc)
    return {"pid": row["pid"], "parent_pid": row["parent_pid"], "create_time": row["create_time"],
            "create_time_utc": dt.datetime.fromtimestamp(row["create_time"], dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "executable": row["executable"], "argv": row["command_line"], "command_line": row["command_line"]}


def _state(line: str) -> dict[str, str]:
    fields = dict(re.findall(r"([A-Za-z_]+)=([0-9a-f,-]+)", line))
    if set(fields) != set(STATE_SOURCE): raise StdioAdapterError("STATE_HASH fields differ from pinned engine source")
    for key in ("gdn", "ple", "tail", "pooled", "kv", "mtp", "stale", "dead", "pooled_full"):
        if re.fullmatch(r"[0-9a-f]{16}", fields[key]) is None: raise StdioAdapterError(f"STATE_HASH {key} is malformed")
    if not fields["L"].isdecimal() or re.fullmatch(r"-?\d+,-?\d+", fields["ple_prev"]) is None:
        raise StdioAdapterError("STATE_HASH L/ple_prev is malformed")
    return fields


def _state_fingerprint(src: dict[str, str]) -> dict[str, Any]:
    domains = {key: src[key] for key in STATE_HASH_DOMAINS}
    metadata = {"L_tokens": int(src["L"]), "ple_prev": [int(x) for x in src["ple_prev"].split(",")]}
    return {"algorithm": "fnv1a64", "phase": "post_gen", "cryptographic": False,
            "domains": domains, "metadata": metadata}


def _stop_ids(manifest: dict[str, Any], contract: dict[str, Any], tokenizer) -> list[int]:
    ids = manifest.get("stdio_protocol_binding", {}).get("stop_token_ids")
    binding = contract.get("stdio_protocol_binding", {})
    actual = tokenizer.encode("<|im_end|>", parse_special=True) + tokenizer.encode("<|endoftext|>", parse_special=True)
    if ids != binding.get("stop_token_ids") or ids != actual or ids != [248046, 248044]:
        raise StdioAdapterError("exact stop IDs are absent or disagree with tokenizer and pinned bridge")
    return list(ids)


def _binary(contract: dict[str, Any]) -> tuple[Path, str, str]:
    b = contract.get("runtime_binary_binding")
    if not isinstance(b, dict): raise StdioAdapterError("contract has no authorized binary binding")
    path = b.get("path", b.get("binary_path")); digest = b.get("sha256", b.get("binary_sha256"))
    source = b.get("source_sha", b.get("source_sha256"))
    if not isinstance(path, str) or not isinstance(digest, str) or not isinstance(source, str):
        raise StdioAdapterError("binary binding requires path, binary SHA-256, and source SHA")
    if digest != BIN_SHA or source != SOURCE_SHA: raise StdioAdapterError("binary/source differs from pinned 6bbc/ffeb")
    binary = Path(path)
    h = hashlib.sha256()
    with binary.open("rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    if h.hexdigest() != digest: raise StdioAdapterError("runtime binary changed after contract binding")
    return binary, digest, source


class _Engine:
    def __init__(self, name, config, exe, env, run42, psutil, tokenizer, monitor, session_wait):
        self.name, self.config, self.exe = name, config, exe
        self.run42, self.psutil, self.tokenizer, self.monitor = run42, psutil, tokenizer, monitor
        self.session_wait = session_wait
        self.argv = [str(exe), "--serve", *config["args"]]
        self.stdout_path = Path(config["log"]).with_name("stdout.raw.bin")
        self.stderr_path = Path(config["log"])
        self.command_path = Path(config["log"]).with_name("commands.jsonl")
        for p in (self.stdout_path, self.stderr_path, self.command_path):
            if p.exists(): raise StdioAdapterError(f"engine receipt exists; refusing overwrite: {p}")
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout_file = self.stdout_path.open("xb", buffering=0)
        self.stderr_file = self.stderr_path.open("xb", buffering=0)
        self.command_file = self.command_path.open("x", encoding="utf-8", newline="\n", buffering=1)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(self.argv, cwd=config["cwd"], env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, bufsize=0, creationflags=flags)
        except BaseException:
            self.stdout_file.close(); self.stderr_file.close(); self.command_file.close(); raise
        try:
            self.tree = _EngineTree(self, run42, psutil)
            self.owner = _Owner(self.proc, self.tree, run42, psutil, monitor)
            self.lines: queue.Queue[bytes | None] = queue.Queue(maxsize=512)
            self.read_error = None; self.lock = threading.RLock(); self.stderr_lines = []; self.states = []
            self.stop_readers = threading.Event()
            self.out_thread = threading.Thread(target=self._read_out, daemon=True)
            self.err_thread = threading.Thread(target=self._read_err, daemon=True)
            self.out_thread.start(); self.err_thread.start(); monitor.attach(self.owner, self.tree)
        except BaseException as cause:
            cleanup_error = None
            try: self._stop_unadopted_exact(run42, psutil)
            except BaseException as exc: cleanup_error = exc
            finally:
                for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                    try: stream.close()
                    except OSError: pass
                for file in (self.stdout_file, self.stderr_file, self.command_file):
                    try: file.close()
                    except OSError: pass
            if cleanup_error is not None:
                raise StdioAdapterError(f"engine adoption failed; exact cleanup failed: {cleanup_error}") from cause
            raise

    def _stop_unadopted_exact(self, run42, psutil):
        if self.proc.poll() is not None: return
        expected_exe = os.path.normcase(os.path.realpath(self.exe))
        def current(): return run42._process_record(psutil.Process(self.proc.pid))
        row = current()
        if (row["pid"] != self.proc.pid or row["executable"] != expected_exe or row["command_line"] != self.argv):
            raise StdioAdapterError("startup failed and exact PID/executable/argv no longer match; process left untouched")
        psutil.Process(self.proc.pid).terminate()
        try: self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            row = current()
            if row["executable"] != expected_exe or row["command_line"] != self.argv:
                raise StdioAdapterError("startup cleanup identity changed; refusing to kill")
            psutil.Process(self.proc.pid).kill(); self.proc.wait(timeout=5)

    def _read_out(self):
        total = 0
        try:
            while raw := self.proc.stdout.readline(LINE_MAX + 1):
                if len(raw) > LINE_MAX or not raw.endswith(b"\n"): raise StdioAdapterError("stdout line exceeds 1 MiB")
                total += len(raw)
                if total > LOG_MAX: raise StdioAdapterError("stdout log exceeds 128 MiB")
                self.stdout_file.write(raw)
                while not self.stop_readers.is_set():
                    try: self.lines.put(raw, timeout=.25); break
                    except queue.Full: continue
        except BaseException as e: self.read_error = e
        finally:
            try: self.lines.put_nowait(None)
            except queue.Full: pass

    def _read_err(self):
        total = 0
        try:
            while raw := self.proc.stderr.readline(LINE_MAX + 1):
                if len(raw) > LINE_MAX or not raw.endswith(b"\n"): raise StdioAdapterError("stderr line exceeds 1 MiB")
                total += len(raw)
                if total > LOG_MAX: raise StdioAdapterError("stderr log exceeds 128 MiB")
                self.stderr_file.write(raw)
                text = raw.rstrip(b"\r\n").decode("utf-8", errors="strict")
                if "STATE_HASH " in text or "conversation cache:" in text:
                    with self.lock:
                        if "STATE_HASH " in text: self.states.append(_state(text.split("STATE_HASH ", 1)[1]))
                        self.stderr_lines.append(text)
        except BaseException as e: self.read_error = e

    def _line(self, deadline):
        while True:
            left = deadline - time.monotonic()
            if left <= 0: raise StdioAdapterError("stdio operation deadline expired")
            try: raw = self.lines.get(timeout=min(left, .5))
            except queue.Empty:
                if self.read_error: raise StdioAdapterError(f"stdio reader failed: {self.read_error}") from self.read_error
                if self.proc.poll() is not None: raise StdioAdapterError(f"engine exited {self.proc.returncode}")
                self.monitor.require(); continue
            if raw is None: raise StdioAdapterError("engine stdout closed before a protocol result")
            return raw.rstrip(b"\r\n").decode("utf-8", errors="strict")

    def send(self, command):
        if "\n" in command or "\r" in command or "\0" in command: raise StdioAdapterError("command is not one safe line")
        self.command_file.write(json.dumps({"utc": _utc(), "pid": self.proc.pid, "command": command,
            "sha256": _sha(command.encode())}, ensure_ascii=False) + "\n")
        self.command_file.flush(); os.fsync(self.command_file.fileno())
        self.proc.stdin.write(command.encode() + b"\n"); self.proc.stdin.flush()

    def ready(self, deadline, max_context):
        info={}
        while True:
            line = self._line(deadline)
            if line.startswith("INFO "):
                for item in line.split()[1:]:
                    key, sep, value = item.partition("=")
                    if sep: info[key] = int(value) if value.lstrip("-").isdigit() else value
                continue
            if line.startswith("ERR ") or line.startswith("FATAL"): raise StdioAdapterError(f"engine startup: {line}")
            if line.startswith("READY "):
                parts = line.split()
                if len(parts) < 3 or parts[2] != "stop" or int(parts[1]) < max_context:
                    raise StdioAdapterError(f"READY context/STOP mismatch: {line}")
                return {"line": line, "context": int(parts[1]), "info": info}

    def generate(self, req, stop_ids, deadline):
        ids = req["prompt_token_ids"]; log_start = len(self.stderr_lines)
        before = len(self.states); sample = req["sampling"]
        self.send(f"GEN {req['max_tokens']} temperature={sample['temperature']} top_p={sample['top_p']} seed={sample['seed']} " + ",".join(map(str, ids)))
        emitted=[]; done=None
        while done is None:
            line=self._line(deadline)
            if line.startswith("T "):
                token=line[2:]
                if not token.isdecimal(): raise StdioAdapterError(f"invalid T line: {line}")
                emitted.append(int(token))
                if len(emitted)>req["max_tokens"]: raise StdioAdapterError("engine exceeded generation cap")
            elif line.startswith("DONE "):
                done=line.split()
                if len(done)<9: raise StdioAdapterError(f"short DONE line: {line}")
            elif line.startswith(("PP ","RESUME ","INFO ")): continue
            elif line.startswith(("ERR ","BADM","FATAL")): raise StdioAdapterError(f"GEN failed: {line}")
            else: raise StdioAdapterError(f"unexpected GEN line: {line[:160]}")
        if int(done[1])!=len(emitted) or int(done[2])!=len(ids): raise StdioAdapterError("DONE token/prompt count mismatch")
        finish=done[5]
        if finish not in ("stop","length"): raise StdioAdapterError(f"unsupported finish reason: {finish}")
        visible=list(emitted)
        if finish=="stop":
            if not stop_ids or visible[-1] not in stop_ids: raise StdioAdapterError("stop ID binding is missing or disagrees with T stream")
            visible.pop()
        text=self.tokenizer.decode(visible,errors="strict"); obj=kv.strict_json_object(text)
        until=time.monotonic()+1.0
        fingerprint=None
        while time.monotonic()<until:
            with self.lock:
                if len(self.states)>before: fingerprint=_state_fingerprint(self.states[before]); break
            if self.read_error: raise StdioAdapterError(f"stdio reader failed: {self.read_error}") from self.read_error
            self.monitor.require(); time.sleep(.02)
        result={k:req[k] for k in ("request_id","prompt_sha256","prompt_token_ids","max_tokens","sampling")}
        result.update({"raw_output_utf8":text,"emitted_token_ids":emitted,"finish_reason":finish,
            "prompt_cache_entries":req["prompt_cache"],"stop_token_ids":list(stop_ids),
            "state_fingerprint":fingerprint,
            "state_fingerprint_source":"pinned engine STATE_HASH after GEN" if fingerprint else None,
            "visible_json":obj,"done_line":" ".join(done),
            "prompt_ms":float(done[3]),"decode_ms":float(done[4]),"prompt_tokens_reused":int(done[8]),
            "stderr_protocol_lines":list(self.stderr_lines[log_start:])})
        return result

    def protocol_session(self, action, path, deadline, wait_cap):
        self.send(("SAVE " if action=="save" else "RESTORE ")+str(path)); lines=[]; progress={}; result=None
        phase_deadline=deadline
        while result is None:
            line=self._line(phase_deadline); lines.append(line); parsed=kv.parse_session_line(line,wait_cap)
            if parsed["kind"]=="progress":
                total=parsed["total"]
                if parsed["done"]<progress.get(total,-1): raise StdioAdapterError("SESSION progress moved backwards")
                progress[total]=parsed["done"]
                phase_deadline=min(deadline,time.monotonic()+wait_cap)
            elif parsed["kind"]=="wait": phase_deadline=min(deadline,time.monotonic()+parsed["seconds"]+5)
            elif parsed["kind"]=="result":
                if parsed["action"]!=action: raise StdioAdapterError("SAVE/RESTORE result verb mismatch")
                result=parsed
        if not progress or any(k!=v for k,v in progress.items()): raise StdioAdapterError("SESSION did not reach file end")
        return lines,result

    def close(self):
        return self.owner.close()

    def finish_logs(self):
        self.stop_readers.set()
        for s in (self.proc.stdin,self.proc.stdout,self.proc.stderr):
            try:s.close()
            except OSError:pass
        self.out_thread.join(5); self.err_thread.join(5)
        for f in (self.stdout_file,self.stderr_file): f.flush(); os.fsync(f.fileno()); f.close()
        self.command_file.flush(); os.fsync(self.command_file.fileno()); self.command_file.close()
        if self.out_thread.is_alive() or self.err_thread.is_alive(): raise StdioAdapterError("stdio reader did not close")


class _EngineTree:
    def __init__(self, engine, run42, psutil):
        self.engine,self.run42,self.psutil=engine,run42,psutil
        exe=os.path.normcase(os.path.realpath(engine.exe))
        self.delegate=run42.OwnedServiceTree(engine.proc,psutil,exe,exe,"<direct-cpp-engine>",[])

    def owner_rows(self):
        state=self.delegate.snapshot(); rows=[]
        try:
            root=_owner(self.run42,self.psutil.Process(self.engine.proc.pid))
            rows.append({"role":"engine","pid":root["pid"],"create_time_utc":root["create_time_utc"],
                "executable":root["executable"],"parent_pid":root["parent_pid"],"argv":root["argv"]})
        except self.psutil.NoSuchProcess: pass
        for old in state["descendants"]:
            try: row=_owner(self.run42,self.psutil.Process(old["pid"]))
            except self.psutil.NoSuchProcess: continue
            if self.run42.OwnedServiceTree._same_process_identity(old,row):
                rows.append({"role":"engine-descendant","pid":row["pid"],"create_time_utc":row["create_time_utc"],
                    "executable":row["executable"],"parent_pid":row["parent_pid"],"argv":row["argv"]})
        if not rows:return []
        return kv.validate_owner_snapshot(rows,rows)

    def snapshot(self):
        base=self.delegate.snapshot()
        rows=self.owner_rows()
        return {**base,"actual_service_child":None,"live_service_children":0,"owner_tree":rows,"direct_cpp_root":True}

    def cleanup_descendants(self): return self.delegate.cleanup_descendants()


class _Owner:
    def __init__(self, engine, tree, run42, psutil, monitor):
        self.engine,self.tree,self.run42,self.psutil,self.monitor=engine,tree,run42,psutil,monitor
        self.pid=engine.proc.pid

    @property
    def returncode(self): return self.engine.proc.poll()

    def close(self):
        if self.engine.proc.poll() is not None: return
        forced=self.monitor.failed is not None
        if not forced:
            try:self.engine.proc.stdin.write(b"QUIT\n"); self.engine.proc.stdin.flush()
            except OSError:forced=True
        try:self.engine.proc.wait(timeout=30 if not forced else 2)
        except subprocess.TimeoutExpired:forced=True
        if self.engine.proc.poll() is None:
            root=self.tree.delegate.root_record; current=self.run42._process_record(self.psutil.Process(self.pid))
            if not self.run42.OwnedServiceTree._same_process_identity(root,current):
                raise StdioAdapterError("PID identity changed; refusing native engine signal")
            self.psutil.Process(self.pid).terminate()
            try:self.engine.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                current=self.run42._process_record(self.psutil.Process(self.pid))
                if not self.run42.OwnedServiceTree._same_process_identity(root,current):
                    raise StdioAdapterError("PID identity changed before exact kill")
                self.psutil.Process(self.pid).kill(); self.engine.proc.wait(timeout=5)


def _stable_file(path: Path, max_bytes: int) -> dict[str, Any]:
    before = path.lstat()
    attrs = getattr(before, "st_file_attributes", 0)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise StdioAdapterError("session seed is not a regular single-link non-reparse file")
    if before.st_size <= 0 or before.st_size > max_bytes:
        raise StdioAdapterError("session seed size is outside the contract file cap")
    digest = hashlib.sha256(); size = 0
    with path.open("rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise StdioAdapterError("session seed identity changed while opening")
        while chunk := stream.read(8 << 20): digest.update(chunk); size += len(chunk)
        ended = os.fstat(stream.fileno())
    after = path.lstat()
    ident = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_nlink)
    if ident(before) != ident(ended) or ident(before) != ident(after) or size != before.st_size:
        raise StdioAdapterError("session seed changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest(), "file_id": f"{before.st_dev}:{before.st_ino}",
            "mtime_ns": before.st_mtime_ns, "regular_single_link_no_reparse": True}


def _completion(engine, request, stop_ids, deadline, checkpoint, label):
    checkpoint(label + "/before", engine.tree.owner_rows())
    result = engine.generate(request, stop_ids, deadline)
    checkpoint(label + "/after", engine.tree.owner_rows())
    return result


class HeteroKvStdioAdapter:
    """Concrete, opt-in private C++ stdio adapter. Import alone opens no engine or resource monitor."""
    api_version = API_VERSION

    def __init__(self):
        self._run42 = None; self._psutil = None; self._active = []; self._continuous_checkpoints = []

    def _start(self, name, configs, contract, tokenizer, deadline):
        config_path = Path(configs[name]["path"])
        raw = config_path.read_bytes()
        if _sha(raw) != configs[name]["sha256"]: raise StdioAdapterError(f"{name} config changed")
        config = kv.strict_json_loads(raw)
        exe, _, _ = _binary(contract)
        if Path(config["exe"]).resolve() != exe.resolve(): raise StdioAdapterError(f"{name} config binary path differs from v2 binding")
        if Path(config["tokenizer"]).resolve() != Path(contract["stdio_protocol_binding"]["tokenizer_directory"]).resolve():
            raise StdioAdapterError(f"{name} config tokenizer directory differs from v2 binding")
        if config.get("hetero_transport") != "private stdio StrataEngine; no HTTP listener":
            raise StdioAdapterError(f"{name} is not private stdio")
        log = Path(config["log"])
        if not log.is_absolute(): raise StdioAdapterError(f"{name} log path must be absolute")
        log.parent.mkdir(parents=True, exist_ok=True)
        monitor_path = log.with_name("resource-owner.jsonl")
        budget = contract["resource_budget"]
        monitor = self._run42.ResourceMonitor(monitor_path, self._psutil,
            minimum_physical=budget["global_physical_floor_bytes"], minimum_commit=budget["global_commit_floor_bytes"])
        try: monitor.start()
        except BaseException:
            try: monitor.close()
            except BaseException: pass
            raise
        engine = None
        try:
            engine = _Engine(name, config, exe, dict(config["env"]), self._run42, self._psutil,
                             tokenizer, monitor, budget["total_arm_deadline_seconds"])
            self._active.append(engine)
            ready = engine.ready(deadline, contract["context"]["limit_tokens"])
            info = ready["info"]
            expected_cache = 4096 if name == "ram-pair" else 0
            for field, expected in (("context", contract["context"]["limit_tokens"]), ("kv", "int8"),
                                    ("pool_workers", 10), ("conversation_cache_mib", expected_cache),
                                    ("engine", contract["runtime_binary_binding"]["engine_version"])):
                if info.get(field) != expected: raise StdioAdapterError(f"{name} INFO {field}={info.get(field)!r}, expected {expected!r}")
            return engine
        except BaseException:
            if engine is not None:
                try: self._close(engine)
                except BaseException: pass
            else:
                try: monitor.close()
                except BaseException: pass
            raise

    def _close(self, engine):
        failure = None
        try: engine.close()
        except BaseException as exc: failure = exc
        try: engine.tree.cleanup_descendants()
        except BaseException as exc:
            if failure is None: failure = exc
        try: engine.finish_logs()
        except BaseException as exc:
            if failure is None: failure = exc
        try: engine.monitor.close()
        except BaseException as exc:
            if failure is None: failure = exc
        try: self._collect_monitor(engine.monitor.path)
        except BaseException as exc:
            if failure is None: failure = exc
        try: remaining = engine.tree.owner_rows()
        except BaseException as exc:
            remaining = [{"pid": engine.proc.pid, "identity_unknown": True}]
            if failure is None: failure = exc
        if not remaining and engine in self._active: self._active.remove(engine)
        elif remaining and failure is None: failure = StdioAdapterError("owned engine descendants remain after cleanup")
        if failure is not None: raise failure
        if engine.proc.returncode != 0: raise StdioAdapterError(f"owned {engine.name} exited {engine.proc.returncode}")

    def _collect_monitor(self, path):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("record_type") != "resource_sample": continue
            if row.get("gate_ok") is not True or row.get("errors"):
                raise StdioAdapterError("continuous resource/owner monitor recorded a failed or unknown sample")
            tree = row.get("owned_service_tree") or {}
            owners = tree.get("owner_tree")
            if not owners: continue  # the first sampler tick may precede Popen/attach
            observed = row.get("observed_at_utc")
            try: stamp = dt.datetime.fromisoformat(observed)
            except (TypeError, ValueError): raise StdioAdapterError("resource monitor timestamp is invalid") from None
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise StdioAdapterError("resource monitor timestamp must include a timezone")
            normalized = stamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            mem = {"status": "pass", "physical_available_bytes": row.get("physical_available_bytes"),
                   "commit_available_bytes": row.get("commit_available_bytes"), "source": row.get("memory_source"),
                   "errors": []}
            self._continuous_checkpoints.append({"label": "continuous-owner-monitor", "resource": mem,
                "owner_tree": owners, "observed_at_utc": normalized,
                "monitor_observed_at_utc": observed, "monitor_source": str(path)})

    def _operate(self, engine, action, path, deadline, contract):
        size = path.stat().st_size if action == "restore" else 0
        if action == "save": size = contract["snapshot_paths"]["max_file_bytes"]
        wait_cap = min(3600, 60 + (size >> 22), contract["resource_budget"]["total_arm_deadline_seconds"])
        lines, result = engine.protocol_session(action, path, deadline, wait_cap)
        return {"action": action, "protocol_lines": lines, "result": result}

    @staticmethod
    def _response_row(request, result):
        row = dict(result)
        row.update({"request_id": request["request_id"], "prompt_sha256": request["prompt_sha256"],
                    "prompt_token_ids": request["prompt_token_ids"], "max_tokens": request["max_tokens"],
                    "sampling": request["sampling"], "prompt_cache_entries": request["prompt_cache"]})
        return row

    @staticmethod
    def _require_reference(conv, turn, actual, references):
        reference = references["conversations"][conv][kv.TURN_IDS.index(turn)]
        if actual["emitted_token_ids"] != reference["emitted_token_ids"]:
            raise StdioAdapterError(f"{conv}/{turn} actual emitted token IDs differ from its uninterrupted reference")
        if actual["finish_reason"] != reference["finish_reason"] or actual["raw_output_utf8"] != reference["raw_output_utf8"]:
            raise StdioAdapterError(f"{conv}/{turn} finish/text differs from its uninterrupted reference")
        a, r = actual.get("state_fingerprint"), reference.get("state_fingerprint")
        if a is not None and r is not None:
            if a["domains"] != r["domains"]:
                raise StdioAdapterError(f"{conv}/{turn} post-GEN FNV hashes differ from its uninterrupted reference")
            if a["metadata"] != r["metadata"]:
                raise StdioAdapterError(f"{conv}/{turn} post-GEN metadata differs from its uninterrupted reference")

    def execute_plan(self, plan: dict[str, Any], request_manifest: dict[str, Any], checkpoint: Callable):
        contract_path = Path(plan.get("contract_path", kv.V2_CONTRACT))
        contract, contract_sha, configs = kv.load_contract(contract_path)
        if (plan.get("execution_contract_version") != 2 or plan.get("contract_sha256") != contract_sha or
                plan.get("run_id") != contract.get("run_id")):
            raise StdioAdapterError("plan is not bound to the root-reviewed v2 contract")
        if plan.get("config_bindings") != configs:
            raise StdioAdapterError("plan config bindings differ from the v2 contract")
        requests = kv.validate_request_manifest(request_manifest, contract, contract_sha)
        tokenizer_dir = Path(contract["stdio_protocol_binding"]["tokenizer_directory"])
        tokenizer = _tokenizer(tokenizer_dir, contract["stdio_protocol_binding"])
        stop_ids = _stop_ids(request_manifest, contract, tokenizer)
        run42 = _helpers(); self._run42 = run42
        import psutil
        self._psutil = psutil
        run_root = Path(contract["runtime_output_directory"])
        if not run_root.is_absolute() or run_root.exists(): raise StdioAdapterError("v2 runtime output root must be new and absolute")
        run_root.parent.mkdir(parents=True, exist_ok=True); run_root.mkdir()
        events_path = run_root / "execution-events.jsonl"
        events = events_path.open("x", encoding="utf-8", newline="\n", buffering=1)
        checkpoints = []
        raw_checkpoint = checkpoint
        def checkpoint(label, owners, additional_bytes=0, additional_disk_bytes=0):
            row = raw_checkpoint(label, owners, additional_bytes, additional_disk_bytes)
            checkpoints.append(row)
            return row
        max_s = contract["resource_budget"]["total_arm_deadline_seconds"]
        deadline = time.monotonic() + max_s
        references = {"schema_version": 2, "contract_sha256": contract_sha,
            "request_manifest_sha256": plan.get("request_manifest_sha256"),
            "stdio_protocol_binding": contract["stdio_protocol_binding"], "conversations": {}}
        actual_ram = {"A": [], "B": []}; ram_events = []; skipped = []; retained = {"A": 0, "B": 0}
        def record(kind, data):
            events.write(json.dumps({"utc": _utc(), "kind": kind, **data}, ensure_ascii=False,
                                     sort_keys=True, allow_nan=False) + "\n")
            events.flush(); os.fsync(events.fileno())
        def generate(engine, conv, turn, phase):
            request = requests[conv][turn]
            result = _completion(engine, request, stop_ids, deadline, checkpoint, f"{phase}/{conv}/{turn}")
            response = self._response_row(request, result)
            record("generation", {"phase": phase, "conversation": conv, "turn": turn, "result": response})
            return response
        try:
            for conv in ("A", "B"):
                checkpoint(f"references/{conv}/prelaunch", [])
                engine = self._start(f"reference-{conv}", configs, contract, tokenizer, deadline)
                try:
                    rows = [generate(engine, conv, turn, "reference") for turn in kv.TURN_IDS]
                    references["conversations"][conv] = rows
                finally: self._close(engine)
            if not references["request_manifest_sha256"]: raise StdioAdapterError("exact request manifest SHA is absent from the plan")
            reference_bindings = kv.validate_reference_manifest(references, requests, contract, contract_sha,
                                                                 references["request_manifest_sha256"])
            _exclusive(run_root / "references.json", _json(references))

            sequence = contract["phases"][1]["sequence"]
            checkpoint("ram/prelaunch", [])
            engine = self._start("ram-pair", configs, contract, tokenizer, deadline)
            last_conv = None
            try:
                for label in sequence:
                    conv, turn = label.split("-", 1); turn = "warmup" if turn == "warm" else turn
                    if turn not in kv.TURN_IDS: raise StdioAdapterError(f"unknown RAM request sequence item: {label}")
                    response = generate(engine, conv, turn, "ram-pair")
                    actual_ram[conv].append(response); self._require_reference(conv, turn, response, references)
                    for line in response.get("stderr_protocol_lines", []):
                        if "conversation cache:" not in line: continue
                        if any(word in line for word in ("skip parking", "discard", "allocation failed", " evicted ")):
                            skipped.append(line); continue
                        park = kv.RAM_PARK_RE.fullmatch(line)
                        restore = kv.RAM_RESTORE_RE.fullmatch(line)
                        if park:
                            if last_conv not in ("A", "B"): raise StdioAdapterError("RAM park line has no preceding conversation")
                            tokens, elapsed, slots, cache_bytes, evictions, snapshot_bytes, reused_bytes = park.groups()
                            event = {"tier": "ram-hot", "conversation": last_conv, "action": "park",
                                "cache_bytes": int(cache_bytes), "snapshot_bytes": int(snapshot_bytes),
                                "admitted": True, "fallback": False, "log_line": line,
                                "request_sequence_index": sequence.index(label)}
                            ram_events.append(event); retained[last_conv] = int(snapshot_bytes)
                        elif restore:
                            tokens, source, elapsed, slots, cache_bytes = restore.groups()
                            ram_events.append({"tier": "ram-hot", "conversation": conv, "action": "restore",
                                "cache_bytes": int(cache_bytes), "reused_prefix_tokens": int(tokens),
                                "admitted": True, "fallback": False, "log_line": line,
                                "request_sequence_index": sequence.index(label)})
                        else: skipped.append(line)
                    last_conv = conv
            finally: self._close(engine)
            ram = {"completions": actual_ram, "evidence": {"tier": "ram-hot", "request_sequence": sequence,
                "fallback_or_skip": skipped, "events": ram_events, "repark_retained_bytes": retained}}
            _exclusive(run_root / "ram-pair.json", _json(ram)); record("ram-evidence", ram["evidence"])
            kv.validate_arm_completions(actual_ram, requests, reference_bindings, contract)
            kv.validate_ram_evidence(ram["evidence"], contract)

            disk_dir = Path(contract["snapshot_paths"]["directory"])
            checkpoint("disk/prelaunch", [])
            if disk_dir.exists() or not disk_dir.parent.is_dir():
                raise StdioAdapterError("disk snapshot directory must be absent under an existing parent")
            engine = self._start("disk-pair", configs, contract, tokenizer, deadline)
            disk_ops = []; seeds = {}; expected_tokens = {}
            disk_sequence = contract["phases"][2]["sequence"]
            disk_completions = {"A": [], "B": []}
            snapshot_dir_created = False
            try:
                for conv, save_after in (("A", "A-warm"), ("B", "B-warm")):
                    warm = generate(engine, conv, "warmup", "disk-pair")
                    self._require_reference(conv, "warmup", warm, references)
                    disk_completions[conv].append(warm)
                    reference_warm = references["conversations"][conv][0]
                    expected_tokens[conv] = (reference_warm["state_fingerprint"]["metadata"]["L_tokens"] if reference_warm.get("state_fingerprint")
                                             else len(reference_warm["prompt_token_ids"]) + len(reference_warm["emitted_token_ids"]))
                    path = disk_dir / f"{conv}-seed.bin"
                    if path.exists(): raise StdioAdapterError("SAVE destination already exists; refusing overwrite")
                    if not disk_dir.exists() and not snapshot_dir_created:
                        checkpoint("disk/snapshot-directory/prelaunch", [])
                        disk_dir.mkdir()
                        snapshot_dir_created = True
                    elif not disk_dir.exists(): raise StdioAdapterError("disk snapshot directory disappeared during the arm")
                    elif not snapshot_dir_created: raise StdioAdapterError("disk snapshot directory appeared after fresh absent check")
                    checkpoint(f"disk/save/{conv}/before", engine.tree.owner_rows(), additional_disk_bytes=contract["snapshot_paths"]["max_file_bytes"])
                    op = self._operate(engine, "save", path, deadline, contract)
                    file = _stable_file(path, contract["snapshot_paths"]["max_file_bytes"])
                    if op["result"]["tokens"] != expected_tokens[conv] or op["result"]["bytes"] != file["bytes"]:
                        raise StdioAdapterError(f"{conv} SAVED token/byte counts disagree with the captured reference seed")
                    checkpoint(f"disk/save/{conv}/after", engine.tree.owner_rows())
                    seed = {"conversation": conv, "path": str(path), **file,
                        "readback_sha256": file["sha256"], "after_sha256": file["sha256"],
                        "before_file_id": file["file_id"], "after_file_id": file["file_id"],
                        "before_mtime_ns": file["mtime_ns"], "after_mtime_ns": file["mtime_ns"],
                        "destination_absent_before_save": True, "created_exclusive": True}
                    seeds[conv] = seed
                    disk_ops.append({"tier": "disk-parked-cold", "action": "save", "conversation": conv,
                        "seed_sha256": file["sha256"], "protocol_lines": op["protocol_lines"]})
                    record("disk-save", {"conversation": conv, "seed": seed, "operation": op})
                restore_order=[]
                restore_labels = [label for label in disk_sequence if label.startswith("RESTORE ")]
                if len(restore_labels) != 6: raise StdioAdapterError("contract disk sequence must bind six restores")
                for index, label in enumerate(restore_labels):
                    match = re.fullmatch(r"RESTORE ([AB])-seed\.bin; \1-(formal[123])", label)
                    if not match: raise StdioAdapterError(f"unknown disk request sequence item: {label}")
                    conv, turn = match.groups()
                    path = Path(seeds[conv]["path"]); before = _stable_file(path, contract["snapshot_paths"]["max_file_bytes"])
                    if before["sha256"] != seeds[conv]["sha256"] or before["bytes"] != seeds[conv]["bytes"]:
                        raise StdioAdapterError(f"{conv} seed changed before RESTORE {index + 1}")
                    checkpoint(f"disk/restore/{conv}/{index+1}/before", engine.tree.owner_rows(),
                               additional_bytes=before["bytes"] + (16 << 20) + ((before["bytes"] // (16 << 20) + 1) * 4096) + (1 << 20))
                    op = self._operate(engine, "restore", path, deadline, contract)
                    after = _stable_file(path, contract["snapshot_paths"]["max_file_bytes"])
                    if (after["sha256"] != before["sha256"] or after["bytes"] != before["bytes"] or
                            op["result"]["tokens"] != expected_tokens[conv] or op["result"]["bytes"] != before["bytes"]):
                        raise StdioAdapterError(f"{conv} RESTORE changed seed or returned mismatched token/byte counts")
                    checkpoint(f"disk/restore/{conv}/{index+1}/after", engine.tree.owner_rows())
                    disk_ops.append({"tier": "disk-parked-cold", "action": "restore", "conversation": conv,
                        "seed_sha256": before["sha256"], "seed_sha256_before_restore": before["sha256"],
                        "seed_bytes_before_restore": before["bytes"], "seed_sha256_after_restore": after["sha256"],
                        "seed_bytes_after_restore": after["bytes"], "protocol_lines": op["protocol_lines"]})
                    restore_order.append(conv)
                    response = generate(engine, conv, turn, "disk-pair"); disk_completions[conv].append(response)
                    self._require_reference(conv, turn, response, references)
                    record("disk-restore-generation", {"conversation": conv, "turn": turn,
                        "seed_before": before, "seed_after": after, "operation": op, "result": response})
                disk_evidence = {"tier": "disk-parked-cold", "seed_files":[seeds["A"],seeds["B"]],
                    "save_order":["A","B"], "restore_order":restore_order, "operations":disk_ops}
            finally: self._close(engine)
            disk_bundle = {"completions":disk_completions,"evidence":disk_evidence,"expected_seed_tokens":expected_tokens}
            _exclusive(run_root / "disk-pair.json", _json(disk_bundle))
            kv.validate_arm_completions(disk_completions, requests, reference_bindings, contract)
            kv.validate_disk_evidence(disk_evidence, contract, expected_tokens)
            record("complete", {"status":"adapter_execution_complete_no_claim_beyond_contract","runtime_root":str(run_root)})
            events.flush(); os.fsync(events.fileno()); events.close()
            merged = {row["observed_at_utc"]: row for row in checkpoints + self._continuous_checkpoints}
            return {"references":references,"ram_pair":ram,"disk_pair":disk_bundle,
                "resource_checkpoints":[merged[t] for t in sorted(merged)],
                "arm_elapsed_seconds":max_s-(deadline-time.monotonic())}
        except BaseException as exc:
            try: record("failure", {"error_type":type(exc).__name__,"error":str(exc)})
            except BaseException: pass
            for engine in list(reversed(self._active)):
                try:self._close(engine)
                except BaseException:pass
            events.flush(); os.fsync(events.fileno()); events.close()
            raise

    def owned_process_trees(self):
        return [engine.tree.owner_rows() for engine in self._active]

    def owner_snapshot(self, expected):
        pid = expected[0]["pid"] if expected else None
        for engine in self._active:
            if engine.proc.pid == pid:
                return engine.tree.owner_rows()
        try: self._psutil.Process(pid)
        except (self._psutil.NoSuchProcess, TypeError): return []
        raise StdioAdapterError("PID is live but is not in this adapter's owned tree")

    def stop_exact(self, identity):
        for engine in self._active:
            rows = engine.tree.owner_rows()
            if any(row["pid"] == identity["pid"] for row in rows):
                if identity["pid"] == engine.proc.pid: self._close(engine)
                else: engine.tree.cleanup_descendants()
                return "stopped_exact_owned_identity"
        raise StdioAdapterError("refusing to signal a process outside this adapter's exact owner trees")


def create_adapter():
    return HeteroKvStdioAdapter()
