"""Local token evidence fixtures: no sockets, model files, devices or native engine.

Prepared source only. Run after the coordinator grants a runtime window:
    python -m unittest serve.test_hetero_token_ids -v
The batch fixture exercises Service's parallel branch, not native BT/BDONE transport.
"""
import contextlib
import io
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.frontend import ChatTemplate, Event
from serve.server import (ByteTokenizer, EngineNonFinite, MockEngine, Service,
                          openai_chunks, run_with_mcp)

ROOT = Path(__file__).resolve().parents[1]
EOS = 257
ALIAS_A = 1000


class AliasTokenizer(ByteTokenizer):
    """Two distinct IDs decode to A; visible-text re-encoding loses the distinction."""
    def decode(self, ids, errors="replace"):
        return super().decode([65 if t == ALIAS_A else t for t in ids], errors)


class PlannedEngine(MockEngine):
    def __init__(self, tok, plans, batch=0):
        super().__init__(tok, "", max_context=4096)
        self.plans = iter(plans)
        self.batch = batch
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        for item in next(self.plans):
            if isinstance(item, Exception):
                raise item
            yield item


def service(plans, batch=0):
    tok = AliasTokenizer()
    eng = PlannedEngine(tok, plans, batch)
    return Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


def chunks(svc, **req):
    with contextlib.redirect_stdout(io.StringIO()):
        return list(openai_chunks(svc, {"stream": True, **req}, [9], False, None, 512,
                                  threading.Event()))


def diagnostic_ids(last):
    return last["strata_diagnostics"]["actual_generated_token_ids"]


class TokenEvidence(unittest.TestCase):
    def test_default_off_and_request_cannot_enable_it(self):
        svc = service([[65, EOS]])
        self.assertIs(svc.hetero_capture_token_ids, False)
        out = chunks(svc, hetero_capture_token_ids=True)
        self.assertTrue(all("strata_diagnostics" not in x for x in out if x is not None))

    def test_config_on_still_requires_literal_stream_true(self):
        for stream in (False, None, 1, "true"):
            with self.subTest(stream=stream):
                svc = service([[65, EOS]])
                svc.hetero_capture_token_ids = True
                out = chunks(svc, stream=stream)
                self.assertNotIn("strata_diagnostics", out[-1])

    def test_actual_alias_and_eos_survive_with_no_heartbeat_or_visible_reencoding(self):
        svc = service([[None, ALIAS_A, 66, EOS]])
        svc.hetero_capture_token_ids = True
        out = chunks(svc)
        text = "".join(x["choices"][0]["delta"].get("content", "") for x in out if x)
        self.assertEqual(text, "AB")
        self.assertEqual(diagnostic_ids(out[-1]), [ALIAS_A, 66, EOS])
        self.assertNotEqual(diagnostic_ids(out[-1]), svc.tok.encode(text))
        self.assertEqual(out[-1]["usage"]["completion_tokens"], 3)
        self.assertEqual(sum("strata_diagnostics" in x for x in out if x), 1)

    def test_on_and_off_have_identical_non_diagnostic_chunks(self):
        streams = []
        for capture in (False, True):
            svc = service([[ALIAS_A, 66, EOS]])
            svc.hetero_capture_token_ids = capture
            out = chunks(svc)
            streams.append([{k: v for k, v in x.items()
                             if k not in ("id", "created", "strata_diagnostics")} for x in out])
        self.assertEqual(*streams)

    def test_stop_string_filter_does_not_rewrite_accepted_ids(self):
        svc = service([[ALIAS_A, 66, 67, EOS]])
        svc.hetero_capture_token_ids = True
        out = chunks(svc, stop=["AB"])
        self.assertEqual(diagnostic_ids(out[-1]), [ALIAS_A, 66])
        self.assertEqual(out[-1]["choices"][0]["finish_reason"], "stop")

    def test_zero_token_nonfinite_retry_does_not_add_ids(self):
        svc = service([[None, EngineNonFinite("fixture")], [ALIAS_A, EOS]])
        svc.hetero_capture_token_ids = True
        out = chunks(svc)
        self.assertEqual(svc.engine.calls, 2)
        self.assertEqual(diagnostic_ids(out[-1]), [ALIAS_A, EOS])

    def test_partial_nonfinite_keeps_only_already_accepted_ids(self):
        svc = service([[ALIAS_A, 66, EngineNonFinite("fixture")]])
        svc.hetero_capture_token_ids = True
        out = chunks(svc)
        self.assertEqual(svc.engine.calls, 1)
        self.assertEqual(diagnostic_ids(out[-1]), [ALIAS_A, 66])
        self.assertEqual(out[-1]["choices"][0]["finish_reason"], "length")

    def test_twice_nonfinite_has_no_successful_final_diagnostics(self):
        svc = service([[EngineNonFinite("fixture")], [EngineNonFinite("fixture")]])
        svc.hetero_capture_token_ids = True
        with self.assertRaises(EngineNonFinite):
            chunks(svc)
        self.assertEqual(svc.engine.calls, 2)

    def test_reasoning_wrap_ids_are_excluded_across_continuations(self):
        svc = service([[114, 115, 116, EOS], [65, EOS]])
        with contextlib.redirect_stdout(io.StringIO()):
            out = list(svc.run([9], True, None, 512, {"reasoning_budget_tokens": 2},
                               threading.Event(), capture_token_ids=True))
        done = out[-1][1]
        self.assertEqual(diagnostic_ids(done), [114, 115, 65, EOS])
        self.assertGreater(done["completion_tokens"], len(diagnostic_ids(done)))
        self.assertEqual(svc.engine.calls, 2)

    def test_interleaved_parallel_requests_keep_distinct_id_lists(self):
        svc = service([[None, ALIAS_A, EOS], [None, 66, EOS]], batch=2)
        svc.hetero_capture_token_ids = True
        a = openai_chunks(svc, {"stream": True}, [1], False, None, 512, threading.Event())
        b = openai_chunks(svc, {"stream": True}, [2], False, None, 512, threading.Event())
        with contextlib.redirect_stdout(io.StringIO()):
            next(a)  # role
            next(b)
            self.assertIsNone(next(a))  # both requests inside their own engine iterator
            self.assertIsNone(next(b))
            a_done, b_done = list(a)[-1], list(b)[-1]
        self.assertEqual(diagnostic_ids(a_done), [ALIAS_A, EOS])
        self.assertEqual(diagnostic_ids(b_done), [66, EOS])
        self.assertIsNot(diagnostic_ids(a_done), diagnostic_ids(b_done))

    def test_mcp_rounds_aggregate_actual_ids(self):
        # Stub only the tool worker to keep this fixture synchronous and process-free.
        call = SimpleNamespace(id="c1", name="echo", arguments={})

        class RoundService:
            def __init__(self):
                self.round = 0

            def run(self, *args, **kwargs):
                self.round += 1
                assert kwargs == {"declared_only": True, "capture_token_ids": True}
                if self.round == 1:
                    yield "event", Event("tool_call", call=call)
                yield "done", {"finish": "stop", "completion_tokens": 2,
                                "strata_diagnostics": {"actual_generated_token_ids":
                                                       [ALIAS_A if self.round == 1 else 66, EOS]}}

            def prepare(self, *args, **kwargs):
                return [10], False, 512

        class InlineWorker:
            def __init__(self, target, **kwargs):
                self.target = target

            def start(self):
                self.target()

            def is_alive(self):
                return False

        hub = SimpleNamespace(settings={"max_rounds": 1}, routes=lambda: {},
                              call=lambda *args: {"ok": True, "text": "ok", "chars": 2,
                                                  "ms": 0, "truncated": False})
        with mock.patch("serve.server.threading.Thread", InlineWorker), contextlib.redirect_stdout(io.StringIO()):
            out = list(run_with_mcp(RoundService(), hub, [], None, {}, [9], False, 512, 512, {},
                                    threading.Event(), {"echo"}, capture_token_ids=True))
        self.assertEqual(diagnostic_ids(out[-1][1]), [ALIAS_A, EOS, 66, EOS])
        self.assertEqual(out[-1][1]["completion_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
