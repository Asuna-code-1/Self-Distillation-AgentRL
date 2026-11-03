"""Use the native row converter while retaining the original downloaded files."""
import ast,json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
source=ROOT/'examples/data_preprocess/preprocess_search_r1_dataset.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_single_row')
namespace={'pd':pd,'user_content_prefix':'','system_content':'You are a helpful and harmless assistant.'}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
convert=namespace['process_single_row']
output=ASSETS/'data/searchR1_processed_direct';output.mkdir(parents=True,exist_ok=True)
manifest={}
for split in ('train','test'):
 raw=ASSETS/f'downloads/searchqa-raw/{split}.parquet'
 target=output/f'{split}.parquet'
 if not target.exists():
  frame=pd.read_parquet(raw)
  processed=frame.apply(lambda row:convert(row,split,row.name),axis=1)
  stage=target.with_suffix('.parquet.preparing')
  processed.to_parquet(stage,index=False)
  assert len(processed)==pq.ParquetFile(raw).metadata.num_rows
  stage.rename(target)
 count=pq.ParquetFile(target).metadata.num_rows
 assert 'env_kwargs' in pq.read_schema(target).names
 manifest[split]={'rows':count,'path':str(target),'raw_retained':str(raw)}
 print('NATIVE_QA_DATA_READY',split,count,flush=True)
(output/'preparation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
