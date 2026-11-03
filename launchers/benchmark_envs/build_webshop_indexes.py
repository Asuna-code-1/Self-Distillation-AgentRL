"""Build native Lucene indexes for both released small/full product resources."""
import json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
sys.path.insert(0,str(ROOT/'agent_system/environments/env_package/webshop/webshop'))
from web_agent_site.engine.engine import load_products
from pyserini.search.lucene import LuceneSearcher

def docs(products):
 for product in products:
  option_text=', and '.join(f"{name}: {', '.join(values)}" for name,values in product.get('options',{}).items())
  yield {'id':product['asin'],'contents':' '.join([product['Title'],product['Description'],product['BulletPoints'][0],option_text]).lower(),'product':product}

def build(name,products):
 source=ASSETS/'data/webshop-lucene'/('resources'+name)
 target=ASSETS/'data/webshop-lucene'/('indexes'+name)
 source.mkdir(parents=True,exist_ok=True)
 marker=target/'local-index-manifest.json'
 if not marker.exists():
  with (source/'documents.jsonl').open('w') as handle:
   for doc in docs(products):handle.write(json.dumps(doc)+'\n')
  result=subprocess.run([sys.executable,'-m','pyserini.index.lucene','--collection','JsonCollection','--input',str(source),'--index',str(target),'--generator','DefaultLuceneDocumentGenerator','--threads','2','--storePositions','--storeDocvectors','--storeRaw'],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (target/'index-build.log').write_text(result.stdout)
  counters={key:int(value.replace(',','')) for key,value in re.findall(r' - (indexed|unindexable|empty|skipped|errors):\s+([\d,]+)',result.stdout)}
  searcher=LuceneSearcher(str(target))
  assert counters.get('errors')==0,counters
  assert counters.get('unindexable')==0 and counters.get('skipped')==0,counters
  assert searcher.num_docs==counters['indexed'],counters
  assert searcher.num_docs+counters['empty']==len(products),counters
  marker.write_text(json.dumps({'documents':searcher.num_docs,'source_documents':len(products),'native_lucene_counters':counters,'native_pipeline':True},indent=2)+'\n')
 print('INDEX_READY',name,len(products),flush=True)

small, *_=load_products(str(ASSETS/'data/webshop/items_shuffle_1000.json'),str(ASSETS/'data/webshop/items_ins_v2_1000.json'))
build('_100',small[:100]);build('_1k',small)
full, *_=load_products(str(ASSETS/'data/webshop/items_shuffle.json'),str(ASSETS/'data/webshop/items_ins_v2.json'))
build('_100k',full[:100000]);build('',full)
print('ALL_WEBSHOP_INDEXES_READY',flush=True)
