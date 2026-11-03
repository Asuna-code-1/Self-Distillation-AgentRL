"""Download original benchmark assets; preserve archives and partial downloads."""
import concurrent.futures
from datetime import datetime,timezone
import hashlib,json,os,shutil,threading,time,traceback
from pathlib import Path

ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
os.environ.update(HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='120',TMPDIR=str(ROOT/'tmp'))
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
LOCK=threading.Lock()
STATE={}
def status(job,**fields):
 with LOCK:
  STATE.setdefault(job,{}).update(fields,updated_at=datetime.now(timezone.utc).isoformat())
  (ROOT/'logs/asset-status.json').write_text(json.dumps(STATE,indent=2)+'\n')
  print(job,fields,flush=True)
def download_repo(job,repo,kind,files,target):
 status(job,stage='metadata')
 info=HfApi().repo_info(repo,repo_type=kind,files_metadata=True)
 selected={s.rfilename:s for s in info.siblings if s.rfilename in files}
 if len(selected)!=len(files):raise ValueError(f'Missing upstream files for {repo}: {set(files)-set(selected)}')
 status(job,stage='downloading',revision=info.sha,bytes=sum(s.size or 0 for s in selected.values()))
 target.mkdir(parents=True,exist_ok=True)
 def one(name):
  p=Path(hf_hub_download(repo,filename=name,repo_type=kind,revision=info.sha,local_dir=target))
  entry=selected[name]
  assert p.stat().st_size==entry.size
  lfs=entry.lfs
  expected=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
  digest=hashlib.sha256()
  with p.open('rb') as f:
   while chunk:=f.read(16*1024*1024):digest.update(chunk)
  actual=digest.hexdigest()
  if expected and expected!=actual:raise ValueError('SHA256 mismatch: '+str(p))
  status(job,last_verified=name)
  return {'file':name,'bytes':p.stat().st_size,'sha256':actual,'upstream_lfs_match':expected==actual if expected else None}
 with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,len(files))) as pool:verified=list(pool.map(one,files))
 manifest={'repo':repo,'kind':kind,'revision':info.sha,'files':verified}
 (target/(job+'-download-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 status(job,stage='completed')
 return manifest

def alfworld():
 src=Path('/workspace/fudan/donghaosen/experiment_data/seed/alfworld');dst=ROOT/'data/alfworld'
 status('alfworld',stage='copying_existing_verified_assets')
 def copy(source,target):
  if Path(target).exists():
   if Path(source).stat().st_size!=Path(target).stat().st_size:raise ValueError('Existing file differs: '+str(target))
   return target
  return shutil.copy2(source,target)
 shutil.copytree(src,dst,copy_function=copy,dirs_exist_ok=True)
 games={split:len(list((dst/'json_2.1.1'/split).rglob('game.tw-pddl'))) for split in ['train','valid_seen','valid_unseen']}
 assert all(games.values()),games
 status('alfworld',stage='completed',games=games)

def retrieval():
 download_repo('searchqa-index','PeterJinGo/wiki-18-e5-index','dataset',['part_aa','part_ab'],ROOT/'data/searchR1')
def corpus():
 download_repo('searchqa-corpus','PeterJinGo/wiki-18-corpus','dataset',['wiki-18.jsonl.gz'],ROOT/'data/searchR1')
def qa():
 download_repo('searchqa-raw','PeterJinGo/nq_hotpotqa_train','dataset',['train.parquet','test.parquet'],ROOT/'downloads/searchqa-raw')
def webshop():
 download_repo('webshop','HongbangYuan/webshop','dataset',['items_shuffle_1000.json','items_ins_v2_1000.json','items_human_ins.json','items_shuffle.json','items_ins_v2.json'],ROOT/'data/webshop')
def e5():
 repo='intfloat/e5-base-v2';info=HfApi().model_info(repo)
 names={s.rfilename for s in info.siblings}
 weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
 files=[f for f in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt',weight] if f in names]
 download_repo('searchqa-e5',repo,'model',files,ROOT/'models/e5-base-v2')

def safe(fn):
 try:fn();return True
 except Exception as error:status(fn.__name__,stage='failed',error=str(error));traceback.print_exc();return False
if __name__=='__main__':
 jobs=[alfworld,webshop,qa,e5,retrieval,corpus]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(safe,jobs))
 if not all(results):raise SystemExit(1)
 print('ALL_ORIGINAL_ASSETS_DOWNLOADED_AND_VERIFIED',flush=True)
