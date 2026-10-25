"""GPU-free source contracts; CUDA graph runtime acceptance is separate."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = "404d05be05f1f449cce91d6883dd4a93d0dd4787"


def without_probe(text):
    out, stack = [], []
    for line in text.splitlines(keepends=True):
        if line.strip() == "#if defined(STRATA_VERIFY_FINITE_PROBE)":
            stack.append(True)
        elif stack and re.match(r"\s*#if", line):
            stack.append(False)
        elif stack and line.strip() == "#endif":
            stack.pop()
        elif not stack:
            out.append(line)
    assert not stack
    return "".join(out)


class SourceContracts(unittest.TestCase):
    def test_math_off_exact_base(self):
        current = (ROOT / "src/core/verify.cpp").read_text()
        base = subprocess.check_output(["git", "show", BASE + ":src/core/verify.cpp"], cwd=ROOT).decode()
        self.assertEqual(without_probe(current), without_probe(base))

    def test_capture_is_before_next_layer(self):
        source = (ROOT / "src/core/verify.cpp").read_text()
        scan = source.index("verify_probe::Stage::routed_parts")
        capture = source.index("finite_probe_.capture_route(rv,err)", scan)
        combine = source.index("if (sg_gated_[grp])", capture)
        self.assertLess(scan, capture)
        self.assertLess(capture, combine)
        self.assertIn("rv.ids=ids_+tb*K;rv.weights=w_+tb*K;rv.mixed=mixed_+tb*N", source)
        self.assertIn("rv.cpu=m_ymiss_+(size_t)tb*K*N", source)
        self.assertIn("rv.cpu_input=m_x_+tb*N", source)
        self.assertIn("rv.cpu_ids=m_ids_+tb*K", source)
        self.assertEqual(source.count("finite_probe_.report_route(err)"), 5)
        poll = source[source.index("int Verifier::batch_poll"):source.index("bool Verifier::", source.index("int Verifier::batch_poll"))]
        self.assertIn("if (!finite_probe_.report_route(err)) { b_running_ = false; return -1; }", poll)

    def test_persistent_snapshot_and_no_disabled_cuda(self):
        source = (ROOT / "src/core/verify_probe.cu").read_text()
        init = source[source.index("bool Probe::init"):source.index("bool Probe::begin_graph")]
        self.assertLess(init.index("if(!enabled)return true"), init.index("cudaMalloc"))
        begin = source[source.index("bool Probe::begin_graph"):source.index("bool Probe::scan")]
        self.assertNotIn("cudaMalloc", begin)
        capture = source[source.index("bool Probe::capture_route"):source.index("int64_t Probe::report")]
        self.assertLess(capture.index("if(!device_route_)return true"), capture.index("capture_route_kernel<<<"))
        self.assertNotIn("cudaMalloc", capture)
        self.assertNotIn("Synchronize", capture)
        header = (ROOT / "include/strata/core/verify_probe.hpp").read_text()
        record = header[header.index("struct Record"):header.index("constexpr int kRouteMaxWidth")]
        self.assertNotIn("[", record)

    def test_snapshot_json_contract(self):
        source = (ROOT / "src/core/verify_probe.cu").read_text()
        report = source[source.index("bool Probe::report_route"):source.index("bool Probe::check")]
        keys = re.findall(r'\\"([A-Za-z_][A-Za-z_0-9]*)\\"', report)
        expected = set("schema diagnostic_only timing_may_perturb owner window layer position row_offset local_token slot expert width k gpu_entry gpu_entries gpu_pcie gpu_hit_valid mapped_non_gpu_valid cpu_producer_proven native_q8_valid all_resident device_plan skipped remote ring gu_type down_type counts route_weight input_maxabs_finite input_nonfinite q8_half_ds_nan q8_half_ds_posinf q8_half_ds_neginf path_base".split())
        self.assertEqual(set(keys), expected)
        self.assertEqual(len(keys), len(expected))
        self.assertIn('O_CREAT|O_EXCL|O_WRONLY', source)
        self.assertLess(report.index('base+".parts.f32"'), report.index('base+".json"'))
        self.assertLess(report.index('base+".mapped-input.f32"'), report.index('base+".json"'))
        self.assertIn('if(s.cpu_valid && !exclusive_bytes(base+".mapped-input.f32"', report)
        self.assertIn('if(s.cpu_valid && !exclusive_bytes(base+".mapped-ids.i32"', report)
        self.assertIn('float_number((float)maxabs)', report)


if __name__ == "__main__":
    unittest.main()
