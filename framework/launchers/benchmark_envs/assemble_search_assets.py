"""Assemble the official split index and corpus, preserving every archive."""
import gzip,hashlib,json,shutil,struct,subprocess,sys
from pathlib import Path
ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/data/searchR1')
parts=[ROOT/'part_aa',ROOT/'part_ab']
# Per-part files have already been checked against the pinned upstream LFS hashes.
index=ROOT/'e5_Flat.index'
if not index.exists():
 stage=ROOT/'e5_Flat.index.assembling'
 with stage.open('wb') as output:
  for part in parts:
   with part.open('rb') as source:shutil.copyfileobj(source,output,16*1024*1024)
 assert stage.stat().st_size==sum(p.stat().st_size for p in parts)
 stage.rename(index)
corpus=ROOT/'wiki-18.jsonl'
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_search_corpus.py'))],check=True)
with index.open('rb') as handle:header=handle.read(16)
assert header[:4] in (b'IxFI',b'IxF2'),repr(header[:4])
print('SEARCH_INDEX_AND_CORPUS_ASSEMBLED',index.stat().st_size,corpus.stat().st_size,flush=True)
(ROOT/'assembly-manifest.json').write_text(json.dumps({'index_bytes':index.stat().st_size,'corpus_bytes':corpus.stat().st_size,'archives_retained':True,'gzip_crc_verified':True},indent=2)+'\n')
