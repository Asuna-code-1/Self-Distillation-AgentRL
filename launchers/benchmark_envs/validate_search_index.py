"""Read and query the full production index without an HTTP listener or GPU."""
import json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import faiss,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
ROOT=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
faiss.omp_set_num_threads(2)
index=faiss.read_index(str(ROOT/'data/searchR1/e5_Flat.index'))
assert index.is_trained and index.d==768 and index.ntotal>20_000_000
vector=index.reconstruct(0)
scores,ids=index.search(vector[None],3)
assert np.isfinite(scores).all() and float(scores[0,0])>=float(np.dot(vector,vector))-.001
path=ROOT/'models/e5-base-v2'
tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=AutoModel.from_pretrained(path,local_files_only=True).eval()
inputs=tokenizer(['query: What is the capital of France?'],return_tensors='pt')
with torch.no_grad():
 features=model(**inputs).last_hidden_state
 mask=inputs['attention_mask'].bool()
 pooled=features.masked_fill(~mask[...,None],0.).sum(1)/mask.sum(1)[...,None]
 query=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().astype('float32')
scores,ids=index.search(query,3)
assert np.isfinite(scores).all() and np.all((ids>=0)&(ids<index.ntotal))
corpus=ROOT/'data/searchR1/wiki-18.jsonl'
count=0;tail=b''
with corpus.open('rb') as f:
 while data:=f.read(16*1024*1024):count+=data.count(b'\n');tail=data[-1:]
if tail and tail!=b'\n':count+=1
assert count==index.ntotal,(count,index.ntotal)
report={'full_index_loaded':True,'dimension':index.d,'index_vectors':index.ntotal,'corpus_documents':count,'index_corpus_alignment':True,'self_query_passed':True,'real_e5_query_ids':ids.tolist(),'real_e5_query_scores':scores.tolist(),'gpu_used':False,'http_service_started':False}
(ROOT/'logs/searchqa-full-index-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
