#!/usr/bin/env python3
"""Single-run CPU-only MSVC build and CTest for the native CPU service."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; BENCH=Path(__file__).resolve().parent
RECEIPTS=BENCH/'receipts-02'; BUILD=Path(r'E:\Strata-Hetero-data\build\native-cpu-service-20261011-59')
PINNED_PYTHON=Path(r'E:\Strata-Hetero-data\venv\Scripts\python.exe')
GGML=Path(r'E:\Strata-Hetero-data\source\llama-3cf0325')
OLD_RUN=ROOT/'bench/hetero/20261009-28-native-cpu-pool-build'
OLD_PREFLIGHT=ROOT/'bench/hetero/20261009-48-native-cpu-affinity-observe/receipts-preflight-05/toolchain-preflight.json'
SDK_VERSION='10.0.26100.0'; PHYS_MIN=12*1024**3; COMMIT_MIN=4*1024**3; FREE_DISK_MIN=12*1024**3
TARGETS=['hetero_native_expert','hetero_native_expert_metadata_fixture','hetero_worker_affinity_observation','hetero_native_cpu_service','hetero_native_cpu_service_fixture']
SOURCE_RELS=[
'CMakeLists.txt','src/kernels/cpu/expert.cpp','src/kernels/cpu/pool.cpp','src/kernels/cpu/native_expert.cpp',
'src/kernels/cpu/expert_layout.cpp','src/kernels/cpu/iq_avx2.cpp','src/kernels/cpu/iq_avx512.cpp','src/kernels/cpu/kq_avx2.cpp',
'src/kernels/cpu/kq_avx1.cpp','src/kernels/cpu/pool_affinity_win.hpp','include/strata/kernels/cpu/expert.hpp',
'include/strata/kernels/cpu/pool.hpp','include/strata/kernels/cpu/native_expert.hpp','include/strata/kernels/cpu/expert_layout.hpp',
'include/strata/kernels/cpu/iq_avx2.hpp','include/strata/kernels/cpu/iq_avx512.hpp','include/strata/kernels/cpu/kq_avx2.hpp',
'src/program/generate.cpp','src/hetero/native_expert_bench.cpp','src/hetero/native_expert_fixture.cpp',
'src/hetero/worker_affinity_observation_test.cpp','src/hetero/native_cpu_service.cpp','src/hetero/native_cpu_service_fixture.cpp',
'tools/hetero_native_cpu/CMakeLists.txt','tools/hetero_native_cpu/sha256.hpp','tools/hetero_native_cpu/cli_fixtures.cmake',
'tools/hetero_native_cpu/run_build.ps1','tools/hetero_resources.py','tools/hetero_verify_existing.py',
'bench/hetero/20261009-48-native-cpu-affinity-observe/run_cpu48_build.py',
'bench/hetero/20261009-28-native-cpu-pool-build/cmake-controls-verified.json',
'bench/hetero/20261009-48-native-cpu-affinity-observe/receipts-preflight-05/toolchain-preflight.json',
'bench/hetero/20261011-59-native-cpu-identity-fix/run_cpu59_build.py']
HELPER_PATH=ROOT/'bench/hetero/20261009-48-native-cpu-affinity-observe/run_cpu48_build.py'
spec=importlib.util.spec_from_file_location('cpu48_helpers',HELPER_PATH)
if spec is None or spec.loader is None: raise RuntimeError('cannot load reviewed CPU48 helper source')
helper=importlib.util.module_from_spec(spec); sys.modules[spec.name]=helper; spec.loader.exec_module(helper)
def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def write_new(path,obj): return helper.write_new(path,obj)
def write_bytes_new(path,raw): return helper.write_bytes_new(path,raw)
def stable_file(path): return helper.stable_file(path)
def process_conflicts(psutil): return helper.process_conflicts(psutil)
def git_context(): return helper.git_context()
def run_command(*args,**kwargs): return helper.run_command(*args,**kwargs)
def get_vc_env(*args,**kwargs): return helper.get_vc_env(*args,**kwargs)

def pinned_tools():
    controls=json.loads((OLD_RUN/'cmake-controls-verified.json').read_text(encoding='utf-8'))
    old=json.loads(OLD_PREFLIGHT.read_text(encoding='utf-8'))
    tools={'cmake':controls['cmake'],'ninja':controls['ninja'],'compiler':controls['compiler'],
      'ctest':str(Path(controls['cmake']).parent/'ctest.exe'),
      'vcvars':r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'}
    expected={'cmake':controls['cmake_sha256'],'ninja':controls['ninja_sha256'],'compiler':controls['compiler_sha256'],
      'ctest':old['tool_sha256']['ctest'],'vcvars':old['tool_sha256']['vcvars']}
    actual={k:stable_file(Path(v))[0] for k,v in tools.items()}
    for k,d in expected.items():
        if actual[k]!=d: raise RuntimeError(f'pinned {k} SHA-256 mismatch')
    return tools,expected,actual

def source_records(freeze):
    rows=[]
    for rel in SOURCE_RELS:
        path=(ROOT/rel).resolve(); digest,st=stable_file(path); raw=path.read_bytes()
        if sha(raw)!=digest: raise RuntimeError(f'source changed during snapshot: {rel}')
        target=BENCH/'source-freeze-02'/rel
        if freeze: write_bytes_new(target,raw)
        elif not target.is_file() or stable_file(target)[0]!=digest: raise RuntimeError(f'source differs from frozen snapshot: {rel}')
        rows.append({'repo_relative':rel,'path':str(path),'snapshot_path':str(target),'sha256':digest,'bytes':st['size_bytes']})
    return rows

def resource_initial(label):
    from tools.hetero_resources import memory_snapshot
    m,errors=memory_snapshot(); p=m.get('physical_available_bytes'); c=m.get('commit_available_bytes')
    return {'time_utc':utc(),'label':label,'physical_available_bytes':p,'commit_available_bytes':c,'source':m.get('source'),
      'errors':errors,'gate_ok':not errors and type(p) is int and type(c) is int and p>=PHYS_MIN and c>=COMMIT_MIN}

def sdk_record(env):
    sdk=Path(env.get('WindowsSdkDir','')); version=env.get('WindowsSDKVersion','').strip('\\/')
    if version!=SDK_VERSION: raise RuntimeError(f'Windows SDK pin mismatch: {version!r}')
    expected=Path(r'C:\Program Files (x86)\Windows Kits\10')
    if os.path.normcase(os.path.realpath(sdk))!=os.path.normcase(os.path.realpath(expected)): raise RuntimeError(f'Windows SDK root mismatch: {sdk}')
    bindir=sdk/'bin'/SDK_VERSION/'x64'; items={n:bindir/n for n in ('rc.exe','mt.exe')}
    return {'version':version,'root':str(sdk),'tools':{n:{'path':str(p),'sha256':stable_file(p)[0]} for n,p in items.items()}}

def check_common(obs,label):
    row=obs.sample(label+':start')
    if not row['gate_ok']: raise RuntimeError(f'fresh PSAPI 12/4 GiB gate failed: {row}')
    if not BUILD.parent.is_dir() or shutil.disk_usage(BUILD.parent).free<FREE_DISK_MIN: raise RuntimeError('E: free-space reserve below 12 GiB')
    if BUILD.exists(): raise FileExistsError(f'refusing existing build directory: {BUILD}')
    import psutil
    conflicts=process_conflicts(psutil)
    if conflicts: raise RuntimeError(f'compiler/accelerator compiler already active: {conflicts}')
    git=git_context()
    if git['ggml_head']!='3cf03257f219afbe7334045ff7c6a06ac68c627d': raise RuntimeError('GGML pin mismatch')
    ggml_status=subprocess.check_output(['git','-C',str(GGML),'status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True)
    if ggml_status.strip(): raise RuntimeError(f'GGML worktree is not clean: {ggml_status}')
    return row,git,conflicts

def preflight():
    if os.path.normcase(os.path.realpath(sys.executable))!=os.path.normcase(os.path.realpath(PINNED_PYTHON)) or not sys.dont_write_bytecode:
        raise RuntimeError('run59 requires the pinned main venv interpreter with -B')
    if BUILD.exists(): raise FileExistsError(f'refusing existing build directory: {BUILD}')
    if BENCH.exists():
        allowed={'run_cpu59_build.py','receipts-01','source-freeze'}
        unexpected=[p.name for p in BENCH.iterdir() if p.name not in allowed]
        if unexpected: raise FileExistsError(f'refusing nonempty run directory: {unexpected}')
    else: BENCH.mkdir(parents=False,exist_ok=False)
    (RECEIPTS/'preflight').mkdir(parents=True,exist_ok=False)
    rd=RECEIPTS/'preflight'; obs=helper.ResourceObserver(rd/'resource-samples.jsonl')
    try:
        initial,git,conflicts=check_common(obs,'preflight')
        status=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True)
        write_bytes_new(RECEIPTS/'git-status-before.txt',status.encode())
        rows=source_records(True)
        manifest={'schema_version':1,'status':'current_service_source_frozen','head':git['head'],'ggml_head':git['ggml_head'],
          'files':rows,'note':'Current worktree source bytes, including untracked service files; old CPU48 hashes are not substituted.'}
        manifest_hash=write_new(BENCH/'source-freeze-manifest.json',manifest)
        tools,expected,actual=pinned_tools(); env,vcvars=get_vc_env(rd,dict(os.environ),obs)
        toolset=env.get('VCToolsVersion','').strip('\\/')
        if toolset!='14.51.36231' or env.get('VSCMD_ARG_TGT_ARCH','').casefold()!='x64': raise RuntimeError('MSVC toolset/target pin mismatch')
        sdk=sdk_record(env); versions={}
        for key,argv in (('cmake',[tools['cmake'],'--version']),('ninja',[tools['ninja'],'--version']),
          ('compiler',[tools['compiler'],'/Bv']),('ctest',[tools['ctest'],'--version'])):
            out,meta=run_command(key+'-version',argv,ROOT,env,rd,obs,timeout=30,accepted_exit_codes=(0,2) if key=='compiler' else (0,))
            text=out.decode(errors='replace')
            if key=='compiler' and 'Microsoft (R) C/C++ Optimizing Compiler' not in text: raise RuntimeError('cl /Bv identity mismatch')
            versions[key]={'stdout_tail':text[-4000:],'process':meta}
        report={'schema_version':1,'status':'pass','phase':'source_only_preflight','build_started':False,
          'python_executable':sys.executable,'python_version':sys.version,'python_dont_write_bytecode':sys.dont_write_bytecode,
          'head':git['head'],
          'ggml_head':git['ggml_head'],'resource_gate_initial':initial,'compiler_conflicts':conflicts,'tool_paths':tools,
          'expected_tool_sha256':expected,'actual_tool_sha256':actual,'vcvars':vcvars,'msvc_toolset':toolset,
          'msvc_target_arch':'x64','windows_sdk':sdk,'tool_versions':versions,'source_freeze_manifest_sha256':manifest_hash,
          'source_count':len(rows),'targets':TARGETS,'parallel_jobs':1,'gpu_backends':{'CUDA':'OFF','HIP':'OFF','SYCL':'OFF'},
          'native_blob_opened':False,'identity_opened':False,'service_serve_started':False,'inference_called':False,
          'resource_samples':obs.samples,'physical_available_min_bytes':obs.minimum_phys,'commit_available_min_bytes':obs.minimum_commit}
        write_new(RECEIPTS/'preflight.json',report); obs.close(); print(json.dumps(report,indent=2))
    except BaseException as exc:
        try: obs.close()
        except Exception: pass
        failure={'status':'failed','phase':'source_only_preflight','build_started':False,'no_retry':True,
          'error':f'{type(exc).__name__}: {exc}','failed_utc':utc()}
        if not (RECEIPTS/'preflight-failure.json').exists(): write_new(RECEIPTS/'preflight-failure.json',failure)
        raise

def configure_argv(tools):
    return [tools['cmake'],'-S',str(ROOT/'tools/hetero_native_cpu'),'-B',str(BUILD),'-G','Ninja',
      f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_SYSTEM_VERSION={SDK_VERSION}',
      '-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=OFF','-DSTRATA_HIP_GFX906=OFF','-DSTRATA_ENABLE_SYCL=OFF',
      '-DSTRATA_PREFILL_MMQ=OFF','-DSTRATA_MMQ_KQUANTS=OFF','-DSTRATA_ORCA_Q4KS_MMQ=OFF','-DSTRATA_NATIVE_EXPERTS=ON',
      '-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_PORTABLE=ON','-DSTRATA_ISA_FLOOR=',f'-DSTRATA_GGML_DIR={GGML}',
      f'-DHETERO_NATIVE_GGML_DIR={GGML}']

def build_once():
    if os.path.normcase(os.path.realpath(sys.executable))!=os.path.normcase(os.path.realpath(PINNED_PYTHON)) or not sys.dont_write_bytecode:
        raise RuntimeError('run59 requires the pinned main venv interpreter with -B')
    prepath=RECEIPTS/'preflight.json'
    if not prepath.is_file(): raise RuntimeError('source-only preflight receipt missing')
    pre=json.loads(prepath.read_text(encoding='utf-8'))
    if pre.get('status')!='pass' or pre.get('build_started') is not False: raise RuntimeError('preflight not accepted')
    before=source_records(False)
    frozen=json.loads((BENCH/'source-freeze-manifest.json').read_text(encoding='utf-8'))
    if {r['repo_relative']:r['sha256'] for r in before}!={r['repo_relative']:r['sha256'] for r in frozen['files']}: raise RuntimeError('source changed since freeze')
    rd=RECEIPTS/'execution'; rd.mkdir(parents=True,exist_ok=False); obs=helper.ResourceObserver(rd/'resource-samples.jsonl')
    try:
        initial,git,conflicts=check_common(obs,'build')
        if git['head']!=pre['head'] or git['ggml_head']!=pre['ggml_head']: raise RuntimeError('git/GGML changed since preflight')
        tools,expected,actual=pinned_tools(); env,vcvars=get_vc_env(rd,dict(os.environ),obs)
        if env.get('VCToolsVersion','').strip('\\/')!='14.51.36231': raise RuntimeError('MSVC toolset changed since preflight')
        sdk=sdk_record(env)
        if sdk!=pre['windows_sdk'] or actual!=pre['actual_tool_sha256']: raise RuntimeError('SDK/tool identity changed since preflight')
        configure=configure_argv(tools)
        bools=[x.split('=',1)[1] for x in configure if x.startswith('-DSTRATA_') and x.split('=',1)[0] in {
          '-DSTRATA_ENABLE_CUDA','-DSTRATA_ENABLE_HIP','-DSTRATA_HIP_GFX906','-DSTRATA_ENABLE_SYCL','-DSTRATA_PREFILL_MMQ',
          '-DSTRATA_MMQ_KQUANTS','-DSTRATA_ORCA_Q4KS_MMQ','-DSTRATA_NATIVE_EXPERTS','-DSTRATA_BUILD_TESTS','-DSTRATA_PORTABLE'}]
        if len(bools)!=10 or any(v not in ('ON','OFF') for v in bools): raise RuntimeError(f'bad CMake bool controls: {bools}')
        plan={'status':'ready_to_build','head':git['head'],'ggml_head':git['ggml_head'],'build_dir':str(BUILD),
          'source_sha256':{r['repo_relative']:r['sha256'] for r in before},'source_freeze_manifest_sha256':stable_file(BENCH/'source-freeze-manifest.json')[0],
          'tool_paths':tools,'tool_sha256':actual,'configure_argv':configure,'targets':TARGETS,'parallel_jobs':1,
          'windows_sdk':sdk,'static_runtime':'/MT','portable_isa':'/arch:AVX2','literal_boolean_controls':bools,
          'gpu_backends':{'CUDA':'OFF','HIP':'OFF','SYCL':'OFF'},'native_blob_opened':False,'identity_opened':False,
          'service_serve_started':False,'inference_called':False,'resource_gate_before_configure':initial,
          'compiler_conflicts':conflicts,'preflight_receipt_sha256':stable_file(prepath)[0]}
        write_new(rd/'build-plan.json',plan)
        if BUILD.exists(): raise FileExistsError(f'refusing existing build path {BUILD}')
        BUILD.mkdir(parents=False,exist_ok=False)
        _,cm=run_command('configure',configure,ROOT,env,rd,obs,timeout=120)
        if cm['exit_code']!=0: raise RuntimeError('CMake configure failed')
        build=[tools['cmake'],'--build',str(BUILD),'--verbose','--parallel','1','--target',*TARGETS]
        _,bm=run_command('build',build,ROOT,env,rd,obs,timeout=240)
        if bm['exit_code']!=0: raise RuntimeError('CPU service build failed')
        ctest=[tools['ctest'],'--test-dir',str(BUILD),'--output-on-failure','--parallel','1']
        out,tm=run_command('ctest',ctest,ROOT,env,rd,obs,timeout=150)
        if tm['exit_code']!=0: raise RuntimeError('CTest failed')
        if '100% tests passed out of 4' not in out.decode(errors='replace'): raise RuntimeError('CTest did not prove all four tests passed')
        after=source_records(False); bmap={r['repo_relative']:r['sha256'] for r in before}; amap={r['repo_relative']:r['sha256'] for r in after}
        if bmap!=amap: raise RuntimeError('source changed during build or CTest')
        binaries={}
        for t in TARGETS:
            path=BUILD/(t+'.exe'); digest,st=stable_file(path); binaries[t]={'path':str(path),'sha256':digest,'bytes':st['size_bytes']}
        log=(rd/'build.log').read_text(encoding='utf-8',errors='replace')
        import re
        flags={'mt':bool(re.search(r'(?:^|\s)(?:-MT|/MT)(?:\s|$)',log)),'avx2':'/arch:AVX2' in log}
        if not all(flags.values()): raise RuntimeError(f'build did not prove /MT and AVX2: {flags}')
        result={'schema_version':1,'status':'success','run_id':'native-cpu-service-20261011-59','started_utc':initial['time_utc'],
          'finished_utc':utc(),'head':git['head'],'ggml_head':git['ggml_head'],'tools':tools,'tool_sha256':actual,'vcvars':vcvars,
          'windows_sdk':sdk,'configure_argv':configure,'build_argv':build,'ctest_argv':ctest,
          'processes':{'configure':cm,'build':bm,'ctest':tm},'exit_codes':{'configure':0,'build':0,'ctest':0},
          'targets':TARGETS,'binaries':binaries,'build_flags_verified':flags,'source_sha256_before':bmap,'source_sha256_after':amap,
          'source_unchanged_during_build':True,'source_freeze_manifest_sha256':plan['source_freeze_manifest_sha256'],
          'resource_samples':obs.samples,'physical_available_min_bytes':obs.minimum_phys,'commit_available_min_bytes':obs.minimum_commit,
          'resource_sampling':'PSAPI gates before each command and once per second during owned process trees',
          'gpu_backends':{'CUDA':'OFF','HIP':'OFF','SYCL':'OFF'},
          'run_scope':{'native_blob_opened':False,'identity_opened':False,'service_serve_started':False,
                       'inference_called':False,'real_model_or_expert_kernel_run':False},'ctest_passed_four':True,'no_retry':True}
        write_new(rd/'build-receipt.json',result)
        write_new(rd/'resource-sampler-receipt.json',{'status':'pass','samples':obs.samples,
          'physical_available_min_bytes':obs.minimum_phys,'commit_available_min_bytes':obs.minimum_commit,
          'samples_path':str(rd/'resource-samples.jsonl')})
        obs.close(); print(json.dumps(result,indent=2))
    except BaseException as exc:
        try: obs.close()
        except Exception: pass
        failure={'status':'failed','phase':'configure_build_ctest','no_retry':True,'error':f'{type(exc).__name__}: {exc}',
          'failed_utc':utc(),'build_dir':str(BUILD),'old_cpu48_build_and_receipts_touched':False}
        if not (rd/'execution-failure.json').exists(): write_new(rd/'execution-failure.json',failure)
        raise

def main():
    ap=argparse.ArgumentParser(description=__doc__); mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true'); mode.add_argument('--build-once',action='store_true'); args=ap.parse_args()
    if args.preflight: preflight()
    else: build_once()
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'native CPU service run59: {type(exc).__name__}: {exc}',file=sys.stderr); raise SystemExit(2)
