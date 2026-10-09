import ctypes,datetime,json,os,sys,time
from ctypes import wintypes
out,stop=sys.argv[1],sys.argv[2]
class M(ctypes.Structure):
 _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD),('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
class P(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('CommitTotal',ctypes.c_size_t),('CommitLimit',ctypes.c_size_t),('CommitPeak',ctypes.c_size_t),('PhysicalTotal',ctypes.c_size_t),('PhysicalAvailable',ctypes.c_size_t),('SystemCache',ctypes.c_size_t),('KernelTotal',ctypes.c_size_t),('KernelPaged',ctypes.c_size_t),('KernelNonpaged',ctypes.c_size_t),('PageSize',ctypes.c_size_t),('HandleCount',wintypes.DWORD),('ProcessCount',wintypes.DWORD),('ThreadCount',wintypes.DWORD)]
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(M)];k.GlobalMemoryStatusEx.restype=wintypes.BOOL
ps=ctypes.WinDLL('psapi',use_last_error=True);ps.GetPerformanceInfo.argtypes=[ctypes.POINTER(P),wintypes.DWORD];ps.GetPerformanceInfo.restype=wintypes.BOOL
with open(out,'x',encoding='utf-8',buffering=1) as f:
 while True:
  m=M();m.dwLength=ctypes.sizeof(M)
  if not k.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError(ctypes.get_last_error())
  p=P();p.cb=ctypes.sizeof(P)
  if not ps.GetPerformanceInfo(ctypes.byref(p),p.cb):raise ctypes.WinError(ctypes.get_last_error())
  avail=int(m.ullAvailPhys);commit=int((p.CommitLimit-p.CommitTotal)*p.PageSize)
  row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'available_physical_bytes':avail,'committed_bytes':int(p.CommitTotal*p.PageSize),'commit_limit_bytes':int(p.CommitLimit*p.PageSize),'commit_headroom_bytes':commit,'gate_ok':avail>=12*1024**3 and commit>=4*1024**3,'source':'isolated Python stdlib ctypes GlobalMemoryStatusEx + PSAPI GetPerformanceInfo'}
  f.write(json.dumps(row,separators=(',',':'))+'\n')
  if os.path.exists(stop):break
  time.sleep(1)