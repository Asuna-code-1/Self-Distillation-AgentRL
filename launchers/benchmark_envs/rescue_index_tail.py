"""Refetch only slow unfinished range tails, then verify via the normal downloader."""
import concurrent.futures,json,os,time
from pathlib import Path
import requests
ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/data/searchR1')
TOTAL=42949672960
URL='https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/a4d31160a035f30764604f4827cd8f1d0315eb86/part_aa?download=true'
fd=os.open(ROOT/'part_aa.ranged-partial',os.O_RDWR)
parents=[];pieces=[]
for marker in (ROOT/'.range-state/part_aa').glob('*.json'):
 state=json.loads(marker.read_text());start=state['start']+state['bytes'];end=state['end']
 if start>end:continue
 parents.append((marker,state))
 for position in range(start,end+1,1024*1024):pieces.append((position,min(end,position+1024*1024-1)))
def one(piece):
 start,end=piece
 for attempt in range(10):
  try:
   url=URL+f'&tail_rescue={start}&attempt={attempt}'
   r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=60);r.raise_for_status()
   assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{TOTAL}'
   assert len(r.content)==end-start+1
   view=memoryview(r.content);offset=start
   while view:
    written=os.pwrite(fd,view,offset);view=view[written:];offset+=written
   print('TAIL_RANGE_READY',start,end,flush=True)
   return
  except Exception:
   if attempt==9:raise
   time.sleep(2)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(one,pieces))
 os.fsync(fd)
 for marker,state in parents:
  state['bytes']=state['end']-state['start']+1;marker.write_text(json.dumps(state)+'\n')
finally:os.close(fd)
print('TAIL_RESCUED; FULL FILE SHA256 STILL REQUIRED',flush=True)
