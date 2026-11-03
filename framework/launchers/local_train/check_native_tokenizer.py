import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests'))
from transformers import AutoTokenizer
from test_reskill_grpo import trainer,FakeActor
from verl.trainer.ppo.reskill_grpo import GRPOSkillRefinement,sft_payload
from verl.trainer.ppo.rewrite_quality import rewrite_quality_payload

t=trainer();t.tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3.5-2B')
t.config.data.apply_chat_template_kwargs={'enable_thinking':False}
c=GRPOSkillRefinement(t)
steps=[{'observation':'You see a cabinet and a sink. '*80,'response':'check the room '*80,
        'next_observation':'A mug is on the table. '*80,'reward':0.,'valid_action':True} for _ in range(30)]
traces=[{'reward':0.,'invalid_actions':0,'steps':steps} for _ in range(8)]
prompt=c._rewrite_prompt('put a clean mug on the table','wash the mug',traces)
ids=c._prompt_ids(prompt)
assert isinstance(ids,list) and all(isinstance(i,int) for i in ids)
assert 2<len(ids)<=8192
actor=FakeActor(t.tokenizer);c._generate([prompt],actor,384,8192,greedy=True)
actual=actor.requests[0].batch['input_ids'][0];mask=actor.requests[0].batch['attention_mask'][0].bool()
assert actual[mask].tolist()==ids
assert sft_payload(t.tokenizer,[('clean mug','wash then put')],2048,384,c.template_kwargs) is not None
record={'rewrite_prompt':prompt,'raw_rewrite':'<skill>wash then put</skill>',
        'rewrite_quality':{'q':0},'rewrite_token_ids':t.tokenizer.encode('<skill>wash then put</skill>',add_special_tokens=False)}
payload=rewrite_quality_payload(t.tokenizer,[record],8192,384,c.template_kwargs)
assert payload is not None
result={'native_transformers_tokenizer':True,'long_feedback_compacted_prompt_tokens':len(ids),
        'identical_budget_and_collector_tokenization':True,'ordinary_and_quality_payloads_passed':True}
Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/native-tokenizer-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
