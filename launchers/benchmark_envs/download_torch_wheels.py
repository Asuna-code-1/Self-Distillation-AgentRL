"""Prefetch pinned Torch wheels with SHA256 checks; never installs packages."""
import concurrent.futures,html,json,re,sys
from pathlib import Path
from packaging.requirements import Requirement
from packaging.markers import default_environment
import requests
from range_download import download
ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/downloads/wheels')
ROOT.mkdir(parents=True,exist_ok=True)
environment=default_environment();environment.update(python_version='3.10',python_full_version='3.10.20',extra='')

def metadata(name,version):
 r=requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60);r.raise_for_status();return r.json()

def choose(meta):
 files=[f for f in meta['urls'] if f['filename'].endswith('.whl') and
        ('x86_64' in f['filename'] or f['filename'].endswith('none-any.whl')) and
        not any(x in f['filename'] for x in ('win','macos','musllinux','aarch64')) and
        (not re.search(r'-cp\d+',f['filename']) or '-cp310-' in f['filename'] or '-cp39-abi3-' in f['filename'])]
 if not files:raise RuntimeError('No Python 3.10 Linux wheel: '+meta['info']['name'])
 files.sort(key=lambda f: ('manylinux_2_28' not in f['filename'],f['filename']))
 return files[0]

def main():
 torch=metadata('torch','2.10.0')
 selected=[choose(torch)]
 for raw in torch['info']['requires_dist']:
  dep=Requirement(raw)
  if dep.marker and not dep.marker.evaluate(environment):continue
  if dep.name.startswith(('nvidia-','cuda-')) or dep.name=='triton':
   exact=[s.version for s in dep.specifier if s.operator=='==']
   if exact:selected.append(choose(metadata(dep.name,exact[0])))
 for name,version in [('torchvision','0.25.0'),('torchaudio','2.10.0')]:selected.append(choose(metadata(name,version)))
 cpu_url='https://download.pytorch.org/whl/cpu/torch/'
 r=requests.get(cpu_url,timeout=60);r.raise_for_status()
 for href in re.findall(r'href="([^"]+)"',r.text):
  href=html.unescape(href)
  if 'torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl' in href:
   url,sep,checksum=href.partition('#sha256=')
   with requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=60) as response:
    response.raise_for_status();size=int(response.headers['Content-Range'].split('/')[-1])
   selected.append({'filename':'torch-2.10.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl','url':url,'size':size,'digests':{'sha256':checksum}})
   break
 else:raise RuntimeError('Official CPU Torch wheel not found')
 (ROOT/'torch-wheel-manifest.json').write_text(json.dumps([{k:f[k] for k in ('filename','size','digests')} for f in selected],indent=2)+'\n')
 def one(f):
  print('DOWNLOADING',f['filename'],f['size'],flush=True)
  download(f['url'],ROOT/f['filename'],f['size'],f['digests']['sha256'],workers=12)
  print('VERIFIED',f['filename'],flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(one,selected))
 print('ALL_PINNED_TORCH_WHEELS_VERIFIED',flush=True)
if __name__=='__main__':main()
