"""Resumable HTTP ranges, bounded threads, full-file size/SHA validation.

No segment files or prior download attempts are deleted. Completed byte offsets
are persisted per range; only fully validated files are promoted to final paths.
"""
import concurrent.futures,hashlib,json,math,os,threading,time
from pathlib import Path
import requests
NETWORK_LIMIT = threading.BoundedSemaphore(128)


def download(url,target,size,sha256=None,workers=24,chunk_bytes=32*1024*1024):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if target.stat().st_size!=size:raise ValueError('Existing final file has wrong size: '+str(target))
  return verify(target,sha256)
 stage=target.with_name(target.name+'.ranged-partial')
 markers=target.parent/'.range-state'/target.name
 markers.mkdir(parents=True,exist_ok=True)
 fd=os.open(stage,os.O_CREAT|os.O_RDWR,0o644)
 if os.fstat(fd).st_size not in (0,size):raise ValueError('Partial size changed upstream')
 os.ftruncate(fd,size)
 completed=0;lock=threading.Lock();started=time.monotonic()
 def one(index):
  nonlocal completed
  start=index*chunk_bytes;end=min(size,start+chunk_bytes)-1
  mark=markers/f'{index:06}.json'
  try:offset=json.loads(mark.read_text())['bytes'] if mark.exists() else 0
  except (ValueError,KeyError):offset=0
  if not 0<=offset<=end-start+1:raise ValueError('Bad range state')
  with lock:completed+=offset
  for attempt in range(16):
   if start+offset>end:return
   try:
    separator='&' if '?' in url else '?'
    request_url=url+separator+f'range_segment={index}&offset={offset}&attempt={attempt}'
    with NETWORK_LIMIT, requests.get(request_url,headers={'Range':f'bytes={start+offset}-{end}'},stream=True,timeout=(30,120)) as response:
     response.raise_for_status()
     if response.status_code == 200 and size <= 1024*1024:
      full = response.content
      if len(full) != size:raise ValueError('Wrong full response size')
      data = full[start+offset:end+1]
      os.pwrite(fd, data, start+offset)
      with lock:completed+=len(data)
      offset+=len(data)
      mark.write_text(json.dumps({'bytes':offset,'start':start,'end':end})+'\n')
      return
     expected=f'bytes {start+offset}-{end}/{size}'
     if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
      raise ValueError('Server did not honor exact requested range')
     for data in response.iter_content(1024*1024):
      if not data:continue
      if start+offset+len(data)>end+1:raise ValueError('Range response too large')
      view=memoryview(data);position=start+offset
      while view:
       written=os.pwrite(fd,view,position);view=view[written:];position+=written
      offset+=len(data)
      mark.write_text(json.dumps({'bytes':offset,'start':start,'end':end})+'\n')
      with lock:completed+=len(data)
    if start+offset!=end+1:raise IOError('Incomplete range')
    return
   except Exception:
    if attempt==15:raise
    time.sleep(min(30,2**min(attempt,4)))
 def progress():
  while not stop.wait(20):
   with lock:done=completed
   elapsed=max(time.monotonic()-started,1)
   print(json.dumps({'file':target.name,'bytes':done,'total':size,'percent':round(done*100/size,2),'MiB_per_s':round(done/elapsed/2**20,2)}),flush=True)
 stop=threading.Event();thread=threading.Thread(target=progress,daemon=True);thread.start()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,range(math.ceil(size/chunk_bytes))))
  os.fsync(fd)
 finally:
  os.close(fd);stop.set();thread.join()
 digest=verify(stage,sha256)
 stage.rename(target)
 return digest

def verify(path,expected=None):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while data:=stream.read(16*1024*1024):digest.update(data)
 actual=digest.hexdigest()
 if expected and actual!=expected:raise ValueError('SHA256 mismatch: '+str(path))
 return actual
