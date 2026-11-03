"""Non-training benchmark checks. No network listeners are opened by default."""
import argparse,importlib.metadata,json,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')

def alfworld():
 import yaml
 # Select the vendored ALFWorld used by this project's native adapter.
 sys.path.insert(0,str(ROOT/'agent_system/environments/env_package/alfworld'))
 from alfworld.agents.environment import get_environment
 config=yaml.safe_load((ROOT/'agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
 config['dataset']['num_eval_games']=1
 config['general']['use_cuda']=False
 base=get_environment('AlfredTWEnv')(config,train_eval='eval_in_distribution')
 env=base.init_env(batch_size=1)
 try:
  obs,info=env.reset()
  assert len(obs)==1 and 'Your task is to:' in obs[0]
  obs,rewards,done,info=env.step(['look'])
  return {'reset':True,'step':True,'game':info.get('extra.gamefile'),'observation_chars':len(obs[0]),'version':importlib.metadata.version('alfworld')}
 finally:env.close()

def webshop():
 sys.path.insert(0,str(ROOT/'agent_system/environments/env_package/webshop/webshop'))
 from web_agent_site.envs import WebAgentTextEnv
 import spacy
 spacy.load('en_core_web_sm')
 env=WebAgentTextEnv(observation_mode='text',num_products=1000,seed=0)
 obs,info=env.reset(session=0)
 assert obs and env.get_instruction_text()
 obs,reward,done,info=env.step('search[mug]')
 return {'reset':True,'search_step':True,'available_actions':len(env.get_available_actions().get('clickables',[])),'lucene':True,'spacy':True}

def sciworld(allow_port=False):
 import scienceworld
 from scienceworld import ScienceWorldEnv
 import py4j
 files=list(Path(scienceworld.__file__).parent.rglob('*.jar'))
 assert files,'ScienceWorld simulator jar missing'
 for file in files:
  with zipfile.ZipFile(file) as jar:assert jar.testzip() is None
 report={'version':importlib.metadata.version('scienceworld'),'jar_verified':True,'jars':[str(p) for p in files],'reset_step_tested':False,'local_port_started':False}
 if allow_port:
  sys.path.insert(0,str(ROOT))
  from agent_system.scienceworld_protocol import verify_simulator_catalog,resolve_item
  env=ScienceWorldEnv()
  try:
   verify_simulator_catalog(env)
   item=resolve_item('sciworld_0')
   env.load(item['task_name'],item['variation_id'],simplificationStr='')
   obs,info=env.reset()
   obs,reward,done,info=env.step('look around')
   report.update(catalog_verified=True,reset_step_tested=True,local_port_started=True)
  finally:env.close()
 return report

def searchqa():
 import faiss,numpy as np,torch,pyarrow.parquet as pq
 from transformers import AutoTokenizer,AutoModel
 model_path=ASSETS/'models/e5-base-v2'
 tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 model=AutoModel.from_pretrained(model_path,local_files_only=True).eval()
 inputs=tok(['query: What is the capital of France?','passage: Paris is the capital of France.'],padding=True,return_tensors='pt')
 with torch.no_grad():
  output=model(**inputs).last_hidden_state
  mask=inputs['attention_mask'].bool()
  pooled=output.masked_fill(~mask[...,None],0.).sum(1)/mask.sum(1)[...,None]
  vectors=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().astype('float32')
 index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors[1:]);scores,ids=index.search(vectors[:1],1)
 assert int(ids[0,0])==0
 splits={s:pq.ParquetFile(ASSETS/f'data/searchR1_processed_direct/{s}.parquet').metadata.num_rows for s in ('train','test')}
 with (ASSETS/'data/searchR1/wiki-18.jsonl').open() as f:doc=json.loads(next(f))
 assert 'contents' in doc or 'text' in doc
 # The production index is read separately by validate_search_index.py.
 return {'e5_forward_cpu':True,'embedding_dim':vectors.shape[1],'faiss_search':True,'splits':splits,'corpus_first_doc_parsed':True,'http_service_started':False}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('benchmark',choices=['alfworld','webshop','searchqa','sciworld']);parser.add_argument('--allow-local-port',action='store_true');args=parser.parse_args()
 os.environ['ALFWORLD_DATA']=str(ASSETS/'data/alfworld')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 os.environ['OMP_NUM_THREADS']='2'
 functions={'alfworld':alfworld,'webshop':webshop,'searchqa':searchqa,'sciworld':lambda:sciworld(args.allow_local_port)}
 report=functions[args.benchmark]()
 report.update(benchmark=args.benchmark,python=sys.executable)
 target=ASSETS/'logs'/f'{args.benchmark}-smoke.json'
 target.write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str))
if __name__=='__main__':main()
