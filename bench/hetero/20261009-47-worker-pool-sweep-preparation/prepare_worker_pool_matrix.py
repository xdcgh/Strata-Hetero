from pathlib import Path
import datetime,hashlib,json,subprocess
repo=Path(r'C:\Users\DC\Documents\ChatGPT\Strata-Hetero')
root=repo/'bench'/'hetero'
run=root/'20261009-47-worker-pool-sweep-preparation'
template=root/'20261009-41-hetero41-direct-quality'
if (run/'matrix.json').exists():raise FileExistsError(run/'matrix.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
if source_head!='33ff86704d15dfc893bf2daafb1117f49069f1dd' or branch!='codex/45-upstream41-runtime':
 raise SystemExit(f'unexpected preparation base {source_head} / {branch}')
identity_template=json.loads((template/'identity.json').read_text(encoding='utf-8'))
prov_template=json.loads((template/'provenance.json').read_text(encoding='utf-8'))
config_template=json.loads((template/'config.json').read_text(encoding='utf-8'))
fresh_path=template/'fresh-identities.json'
manifest_path=root/'20261009-06-quality-prompts'/'manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest_sha=sha(manifest_path)
pair_path=root/'20261009-39-upstream41-build'/'pair'/'cuda-build-pair-41.json'
pair=json.loads(pair_path.read_text(encoding='utf-8'))
hetero_build=pair['builds'][1]
binary=identity_template['engine']['path'];binary_sha=identity_template['engine']['sha256'];candidate_sha=identity_template['source']['sha']
bridge_path=prov_template['server_bridge']['path'];bridge_sha=prov_template['server_bridge']['sha256']
assert candidate_sha==hetero_build['source_head'] and binary_sha==hetero_build['binary']['sha256']
assert sha(Path(binary))==binary_sha and sha(Path(bridge_path))==bridge_sha
cli_source=(repo/'src'/'program'/'generate.cpp').read_text(encoding='utf-8')
for token in ('else if (a == "--pool-affinity")','v == "auto"','v == "all"','v == "p-cores"'):
 if token not in cli_source:raise SystemExit('affinity CLI support missing: '+token)
source_blobs={path:subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD:'+path],text=True).strip()
 for path in ('src/program/generate.cpp','src/kernels/cpu/pool.cpp','include/strata/kernels/cpu/pool.hpp')}
# Arms: baseline and explicit CLI-supported all/auto candidates; harness-only `none` is excluded.
arms=[('w10-all',10,'all','baseline'),('w04-all',4,'all','candidate'),('w06-all',6,'all','candidate'),('w16-all',16,'all','candidate'),('w06-auto',6,'auto','candidate')]
def normalized(c,include_capture=False):
 x=json.loads(json.dumps(c));x['exe']='@SAME_BINARY@';x['log']='@RUN_LOG@'
 args=x['args'];i=args.index('--pool-workers');args[i+1]='@WORKERS@';j=args.index('--pool-affinity');args[j+1]='@AFFINITY@'
 if not include_capture:x['hetero_capture_token_ids']='@CAPTURE_MODE@'
 return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
all_hashes=[];quality_hashes=[];performance_hashes=[];records=[]
for arm,workers,affinity,role in arms:
 arm_dir=run/'arms'/arm;arm_dir.mkdir(parents=True,exist_ok=True);phase_records={}; phase_configs={}
 for phase,capture in (('quality-on',True),('performance-off',False)):
  runid=f'20261009-47-{arm}-{phase}'
  leaf=arm_dir/phase
  if leaf.exists() and any(leaf.iterdir()):raise FileExistsError(f'nonempty arm destination {leaf}')
  leaf.mkdir(parents=True,exist_ok=True)
  for sub in ('logs','quality','requests','resource'):(leaf/sub).mkdir(exist_ok=True)
  cfg=json.loads(json.dumps(config_template));cfg['exe']=binary;cfg['log']=str(leaf/'logs'/'engine.log');cfg['cwd']=str(repo);cfg['hetero_capture_token_ids']=capture
  args=cfg['args'];wi=args.index('--pool-workers');args[wi+1]=str(workers)
  if '--pool-affinity' in args:args[args.index('--pool-affinity')+1]=affinity
  else:args[wi+2:wi+2]=['--pool-affinity',affinity]
  if 'none' in args:raise SystemExit('harness-only none affinity appeared in planned CLI args')
  cp=leaf/'config.json';cp.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');cfg_sha=sha(cp)
  ident=json.loads(json.dumps(identity_template));ident['config']['sha256']=cfg_sha;ident['config']['scope']='v0.1.41-cpu-pool-full-model-comparison';ident['config']['pool_workers']=workers;ident['config']['pool_affinity']=affinity;ident['config']['token_capture']=capture;ident['config']['arm_id']=arm;ident['acceptance_status']='prepared_not_launched';ident['engine']={'version':'0.1.41','path':binary,'sha256':binary_sha,'size_bytes':hetero_build['binary']['size_bytes'],'source_sha':candidate_sha}
  (leaf/'identity.json').write_text(json.dumps(ident,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');(leaf/'fresh-identities.json').write_bytes(fresh_path.read_bytes())
  prov=json.loads(json.dumps(prov_template));prov['prepared_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();prov['run_id']=runid;prov['arm']=arm;prov['engine_cpp_sha']=candidate_sha;prov['engine_path']=binary;prov['engine_sha256']=binary_sha;prov['server_file_sha256']=bridge_sha;prov['server_python_checkout_sha']=candidate_sha;prov['acceptance_status']='prepared_not_launched';prov['process_receipt']=None
  sc=prov['shared_controls'];sc['quality_prompt_manifest']=str(manifest_path);sc['quality_prompt_manifest_sha256']=manifest_sha;sc['pool_workers']=workers;sc['pool_affinity']=affinity;sc['resident_budget_gib']=20;sc['max_context']=32768;sc['kv']='int8';sc['ple_io']='direct';sc['native_dense_gguf_paths']=config_template['args'][i+1:i+1] if False else sc['native_dense_gguf_paths'];sc['new_v0_1_41_controls']={'STRATA_PREFILL_CPU_SHARE':'0','STRATA_STAGE_PIN':'0'}
  prov['worker_pool']={'workers':workers,'affinity_cli':affinity,'role':role,'semantics':'N worker threads plus participating caller; process/parent affinity is not changed','affinity_actual_applied':'to be verified from engine startup report; do not infer from request flag alone'}
  prov['token_capture']={'enabled':capture,'phase':phase,'timing_policy':'capture-on quality and capture-off performance are separate; never pool their timings'}
  prov['comparison_contract']['worker_pool_arm']=arm;prov['comparison_contract']['pool_workers']=workers;prov['comparison_contract']['pool_affinity']=affinity;prov['comparison_contract']['token_capture_enabled']=capture;prov['comparison_contract']['all_nonarm_runtime_controls_unchanged']=True
  prov['binary_build_evidence']={'pair_summary':str(pair_path),'source_sha':candidate_sha,'binary_path':binary,'binary_sha256':binary_sha,'bridge_path':bridge_path,'bridge_sha256':bridge_sha}
  (leaf/'provenance.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
  phase_cfg_hash=hashlib.sha256(normalized(cfg)).hexdigest();all_hashes.append(phase_cfg_hash)
  (quality_hashes if capture else performance_hashes).append(phase_cfg_hash);phase_configs[phase]=cfg
  preparation={'schema_version':1,'run_id':runid,'status':'prepared_not_launched','config_sha256':cfg_sha,'identity_sha256':sha(leaf/'identity.json'),'provenance_sha256':sha(leaf/'provenance.json'),'fresh_identities_sha256':sha(leaf/'fresh-identities.json'),'worker_count':workers,'affinity':affinity,'token_capture':capture}
  (leaf/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf-8')
 # Ensure quality-on and performance-off configs differ only in log path + ID capture flag for this arm.
 q=phase_configs['quality-on'];p=phase_configs['performance-off']
 if hashlib.sha256(normalized(q)).hexdigest()!=hashlib.sha256(normalized(p)).hexdigest():raise SystemExit(f'non-capture config drift within {arm}')
 records.append({'arm':arm,'role':role,'workers':workers,'affinity':affinity,'quality_on_run_id':f'20261009-47-{arm}-quality-on','performance_off_run_id':f'20261009-47-{arm}-performance-off','quality_config_sha256':sha(arm_dir/'quality-on'/'config.json'),'performance_config_sha256':sha(arm_dir/'performance-off'/'config.json')})
if len(set(all_hashes))!=1 or len(set(quality_hashes))!=1 or len(set(performance_hashes))!=1:raise SystemExit('normalized controls differ across arms')
(run/'preparation.json').write_text(json.dumps({'schema_version':1,'run_id':'20261009-47-worker-pool-sweep-preparation','status':'prepared_not_launched','source_head':source_head,'branch':branch,'current_worktree_status':subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True).splitlines(),'candidate_source_sha':candidate_sha,'candidate_binary_sha256':binary_sha,'bridge_sha256':bridge_sha,'prompt_manifest_sha256':manifest_sha,'normalized_controls_sha256':all_hashes[0],'arms':records,'no_model_or_api_launched':True},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'arms':len(records),'configs':10,'normalized_controls_sha256':all_hashes[0],'quality_capture_sha256':quality_hashes[0],'performance_capture_sha256':performance_hashes[0],'source_blobs':source_blobs},indent=2))