"""Safely extract the JSONL from the upstream gzip-compressed TAR archive."""
import fcntl,gzip,json,shutil,tarfile
from pathlib import Path
ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/data/searchR1')
with (ROOT/'corpus-extraction.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 target=ROOT/'wiki-18.jsonl'
 if not target.exists():
  raw=ROOT/'wiki-18.source.tar'
  previous=ROOT/'wiki-18.jsonl.decompressing'
  if previous.exists() and not raw.exists():
   # Prior gzip decoding completed its CRC check, but revealed a TAR container.
   if not tarfile.is_tarfile(previous):raise ValueError('Unexpected preserved archive format')
   previous.rename(raw)
  if not raw.exists():
   stage=ROOT/'wiki-18.tar.decompressing'
   with gzip.open(ROOT/'wiki-18.jsonl.gz','rb') as source,stage.open('wb') as output:
    shutil.copyfileobj(source,output,16*1024*1024)
   stage.rename(raw)
  with tarfile.open(raw,'r:') as archive:
   members=[m for m in archive.getmembers() if m.isfile() and m.name.endswith('.jsonl')]
   assert len(members)==1,'Expected exactly one JSONL corpus'
   member=members[0]
   stage=ROOT/'wiki-18.jsonl.extracting'
   # Never extract archive paths: write only to our explicit destination.
   with archive.extractfile(member) as source,stage.open('wb') as output:
    shutil.copyfileobj(source,output,16*1024*1024)
   assert stage.stat().st_size==member.size
  with stage.open() as handle:assert 'contents' in json.loads(next(handle))
  stage.rename(target)
 with target.open() as handle:assert 'contents' in json.loads(next(handle))
 (ROOT/'corpus-preparation.json').write_text(json.dumps({'bytes':target.stat().st_size,'source_format':'tar.gz','full_gzip_crc_verified':True,'archive_retained':True,'archive_paths_not_extracted':True},indent=2)+'\n')
 print('WIKIPEDIA_CORPUS_READY',target.stat().st_size,flush=True)
