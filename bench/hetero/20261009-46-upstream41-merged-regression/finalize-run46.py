from pathlib import Path
import datetime, hashlib, json, subprocess
repo = Path(r'C:\Users\DC\Documents\ChatGPT\Strata-Hetero')
run = repo / 'bench' / 'hetero' / '20261009-46-upstream41-merged-regression'
commands_path = run / 'commands.jsonl'
runner = run / 'run-regression.ps1'
results_path = run / 'results.json'
summary_path = run / 'summary.md'
failure_path = run / 'runner-finalization-failure.json'
if any(p.exists() for p in (results_path, summary_path, failure_path)):
    raise FileExistsError('offline finalization outputs already exist')
rows = [json.loads(line) for line in commands_path.read_text(encoding='utf-8').splitlines() if line.strip()]
starts = [x for x in rows if x.get('event') == 'start']
finishes = [x for x in rows if x.get('event') == 'finish']
if len(starts) != len(finishes) or len(starts) != 23:
    raise SystemExit(f'event count mismatch: starts={len(starts)} finishes={len(finishes)}')
finished = {x['suite']: x for x in finishes}
if len(finished) != len(finishes):
    raise SystemExit('duplicate suite finish event')
for item in starts:
    name = item['suite']
    if name not in finished or finished[name].get('exit_code') != 0:
        raise SystemExit(f'command missing successful finish: {name}')
    log = run / 'logs' / f'{name}.log'
    if not log.is_file() or hashlib.sha256(log.read_bytes()).hexdigest() != finished[name]['log_sha256']:
        raise SystemExit(f'log missing or hash mismatch: {name}')
unit = [x for x in finishes if 'tests_run' in x]
fixture = [x for x in finishes if 'fixture_cases' in x]
if len(unit) != 22 or len(fixture) != 1:
    raise SystemExit(f'expected 22 unittest suites and one fixture suite: {len(unit)} / {len(fixture)}')
run_count = sum(int(x['tests_run']) for x in unit)
passed = sum(int(x['tests_passed']) for x in unit)
skipped = sum(int(x['tests_skipped']) for x in unit)
failed = sum(int(x['tests_failed']) for x in unit)
fixture_cases = sum(int(x['fixture_cases']) for x in fixture)
if (run_count, passed, skipped, failed, fixture_cases) != (792, 791, 1, 0, 10):
    raise SystemExit(f'test totals unexpected: {(run_count, passed, skipped, failed, fixture_cases)}')
head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
branch = subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
tracked = subprocess.check_output(['git','-C',str(repo),'diff','--name-only'],text=True).splitlines()
staged = subprocess.check_output(['git','-C',str(repo),'diff','--cached','--name-only'],text=True).splitlines()
preparation = json.loads((run/'preparation-corrected.json').read_text(encoding='utf-8'))
if head != preparation['source_head']:
    raise SystemExit('HEAD changed during tests')
result = {
 'schema_version':1,
 'run_id':'20261009-46-upstream41-merged-regression',
 'status':'all_test_commands_passed_results_reconstructed_after_runner_finalizer_error',
 'source_head':head,'branch':branch,'commands':len(starts),
 'python_test_suites':len(unit),'python_tests_run':run_count,'python_tests_passed':passed,
 'python_tests_skipped':skipped,'python_tests_failed':failed,
 'custom_fixture_suites':len(fixture),'custom_fixture_cases':fixture_cases,
 'all_test_command_exit_codes_zero':True,
 'tracked_diff_paths':tracked,'staged_diff_paths':staged,
 'tracked_source_unchanged':not tracked and not staged,
 'openvino_core_initialized':False,'intel_venv_used':False,
 'download_test_network_calls_mocked':True,
 'execution_scope':'pure tests and fake fixtures only; no compile, model, API, GPU, Core, or real download',
 'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),
 'commands_log_sha256':hashlib.sha256(commands_path.read_bytes()).hexdigest(),
 'suites':finishes,'finalized_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()
}
failure = {
 'schema_version':1,'run_id':result['run_id'],
 'status':'runner_finalization_failed_after_all_commands_passed',
 'source_head':head,'runner_sha256':result['runner_sha256'],
 'tests_started':len(starts),'tests_completed':len(finishes),
 'error':'Write-FinalEvidence ran under StrictMode and accessed tests_run on the custom fixture record, which has fixture_cases only. All test command exit codes were zero; no tests were rerun.',
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()
}
summary = f'''# Upstream41 merged pure regression

All {len(starts)} test commands exited zero. The runner then failed only while aggregating its result because StrictMode accessed `tests_run` on the custom PowerShell fixture entry, which records `fixture_cases`. Results below were reconstructed from the already completed command ledger and suite logs; no tests were rerun.

Source HEAD: `{head}` on `{branch}`. Tracked diff paths: {len(tracked)}; staged diff paths: {len(staged)}.

Python unit totals: {run_count} run, {passed} passed, {skipped} skipped, {failed} failed. Additional PowerShell fixture cases: {fixture_cases}.

Reused run33 command suites: serve (400), calibrate/setup (163), tokenizer (2), 16 Hetero modules (198 on the merged tree), and mocked CUDA-preparation tests (13). Added planner (9) and runtime (7) suites. Commands and complete logs are in `commands.jsonl` and `logs/`.

No compiler, model, API request, GPU, OpenVINO Core/device initialization, or real download was performed.
'''
for path, data in ((failure_path, failure), (results_path, result)):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
with summary_path.open('x', encoding='utf-8', newline='\n') as f:
    f.write(summary)
print(json.dumps({'status':result['status'],'commands':len(starts),'python_tests_run':run_count,
 'python_tests_passed':passed,'python_tests_skipped':skipped,'python_tests_failed':failed,
 'custom_fixture_cases':fixture_cases,'tracked_diff_paths':len(tracked),'staged_diff_paths':len(staged)},separators=(',',':')))
