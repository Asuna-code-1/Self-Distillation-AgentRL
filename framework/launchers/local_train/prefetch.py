"""Fetch native training wheels and official CUDA extensions with checksums."""
import concurrent.futures,hashlib,json,sys
from pathlib import Path
import requests
from packaging.requirements import Requirement
ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/downloads/training')
ROOT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark_envs'))
from range_download import download

def pypi(name,version):
 j=requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=45).json()
 files=[f for f in j['urls'] if f['filename'].endswith('.whl') and
        ('x86_64' in f['filename'] or f['filename'].endswith('none-any.whl')) and
        (('-cp310-' in f['filename']) or '-cp38-abi3-' in f['filename'] or '-cp39-abi3-' in f['filename'] or '-py3-' in f['filename']) and
        not any(x in f['filename'] for x in ('musllinux','macos','win_'))]
 if not files:return
 f=files[0];print('FETCH',f['filename'],f['size'],flush=True)
 download(f['url'],ROOT/f['filename'],f['size'],f['digests']['sha256'],workers=16)
 print('READY',f['filename'],flush=True)

def causal():
 j=requests.get('https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.7.0',timeout=45).json()
 f=next(a for a in j['assets'] if a['name']=='causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl')
 download(f['browser_download_url'],ROOT/f['name'],f['size'],f.get('digest','').removeprefix('sha256:') or None,workers=12)
 print('READY',f['name'],flush=True)
if __name__=='__main__':
 names=[('vllm','0.19.1'),('xformers','0.0.34'),('tilelang','0.1.13'),('flashinfer-cubin','0.6.6'),('flashinfer-python','0.6.6')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  jobs=[pool.submit(pypi,*args) for args in names]+[pool.submit(causal)]
  for f in jobs:f.result()
