import datetime,json,pathlib,sys,time
source=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);stop=pathlib.Path(sys.argv[3]);sys.path.insert(0,str(source))
from tools.hetero_resources import windows_memory_global
with out.open('x',encoding='utf-8',buffering=1) as f:
 while True:
  at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')
  try:
   m=windows_memory_global();row={'sampled_at_utc':at,'physical_total_bytes':m.get('physical_total_bytes'),'physical_available_bytes':m.get('physical_available_bytes'),'commit_limit_bytes':m.get('commit_limit_bytes'),'commit_available_bytes':m.get('commit_available_bytes'),'source':m.get('source'),'error':None}
  except Exception as e: row={'sampled_at_utc':at,'physical_available_bytes':None,'commit_available_bytes':None,'error':type(e).__name__+': '+str(e)}
  f.write(json.dumps(row,separators=(',',':'))+'\n')
  if stop.exists(): break
  time.sleep(1)
