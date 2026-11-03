import json,os,time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from agent_system.environments.prompts.alfworld import ALFWORLD_REASONING_INSTRUCTION,ALFWORLD_ACTION_ONLY_INSTRUCTION,alfworld_prompt_template
root=Path(__file__).resolve().parents[2]
for history in (False,True):
 template=alfworld_prompt_template(history,True)
 assert 'MUST be enclosed' not in template and '<think>' not in template
 assert 'Output only <action>' in template
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
audit=next((assets/'training/checkpoints/reskill_grpo/alfworld').glob('*a03*/skill_refinement/step-1.jsonl'))
groups=json.loads(audit.read_text().splitlines()[-1])['groups']
tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3.5-2B')
prompts=[]
for group in groups[:4]:
 text=group['trajectories'][0]['steps'][0]['next_observation']
 # Remove old sampled action history from this inference-only check by using the
 # first-room observation/admissible-action block from the recorded prompt.
 text=text.replace(ALFWORLD_REASONING_INSTRUCTION,ALFWORLD_ACTION_ONLY_INSTRUCTION)
 assert 'MUST be enclosed' not in text
 prompts.append(tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
assert all(p.endswith('<think>\n\n</think>\n\n') for p in prompts)
model=LLM(model=str(root/'models/Qwen3.5-2B'),dtype='bfloat16',gpu_memory_utilization=.2,max_model_len=2560,max_num_seqs=4,
          enforce_eager=True,enable_chunked_prefill=True,max_num_batched_tokens=4096,
          limit_mm_per_prompt={'image':0,'video':0},gdn_prefill_backend='triton',disable_log_stats=True)
start=time.perf_counter();out=model.generate(prompts,SamplingParams(temperature=0,max_tokens=128),use_tqdm=False)
rows=[{'text':item.outputs[0].text,'tokens':len(item.outputs[0].token_ids)} for item in out]
result={'enable_thinking':False,'action_only_prompt':True,'seconds':time.perf_counter()-start,'outputs':rows,
        'action_tag_fraction':sum(row['text'].strip().startswith('<action>') and '</action>' in row['text'] for row in rows)/len(rows)}
(assets/'logs/training/no-thinking-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert all('<think>' not in row['text'] and '</think>' not in row['text'] for row in rows)
assert all(0<row['tokens']<128 for row in rows)
# Untuned action-format mistakes remain environment feedback for GRPO.
import torch
if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
