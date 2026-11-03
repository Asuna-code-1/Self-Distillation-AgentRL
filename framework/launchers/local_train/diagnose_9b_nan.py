"""Numerics audit and a no-update native Qwen3.5-9B packing/backward probe."""
import json,os,time,traceback
from pathlib import Path
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText,AutoTokenizer
from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.torch_functional import entropy_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor

root=Path(__file__).resolve().parents[2]
dest=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/nan-diagnostic-9b-20261101/native-v2')
dest.mkdir(parents=True,exist_ok=True)
results={}
for dtype in [torch.float32,torch.bfloat16,torch.float64]:
 for case,adv,mask,delta in [('positive_overflow',[1.],[1.],[100.]),('negative_overflow',[-1.],[1.],[100.]),('zero_adv_overflow',[0.],[1.],[100.]),('masked_overflow',[1.,0.],[1.,0.],[0.,100.]),('control',[1.],[1.],[0.]),('empty_mask',[1.],[0.],[0.])]:
  old=torch.full((1,len(adv)),-101.,dtype=dtype)
  current=(old+torch.tensor([delta],dtype=dtype)).requires_grad_()
  loss,*_=compute_policy_loss(old,current,torch.tensor([adv],dtype=dtype),torch.tensor([mask],dtype=dtype),cliprange=.2)
  loss.backward()
  results[str(dtype)+'/'+case]={'loss':str(loss.item()),'loss_finite':bool(torch.isfinite(loss)),'gradient_finite':bool(torch.isfinite(current.grad).all())}
for dtype in [torch.float32,torch.bfloat16]:
 logits=torch.tensor([[1000.,0.,-1000.],[20.,0.,-20.]],dtype=dtype,requires_grad=True)
 ent=entropy_from_logits(logits);ent.sum().backward()
 results[str(dtype)+'/saturated_entropy']={'forward_finite':bool(torch.isfinite(ent).all()),'gradient_finite':bool(torch.isfinite(logits.grad).all())}
(dest/'loss_reproductions.json').write_text(json.dumps(results,indent=2)+'\n')
print('CPU_LOSS_REPRODUCTIONS',json.dumps(results),flush=True)
import swanlab
swanlab.login(api_key=Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip())
swanlab.init(project='agentic_alfworld',experiment_name='nan_diagnostic_9b_no_optimizer_'+str(int(time.time())),mode='online',logdir='/data/fd/donghaosen/Self-Distillation-AgentRL/swanlab',config={'purpose':'native packing and backward diagnostic; no optimizer, no training update','model':'Qwen3.5-9B','use_remove_padding':True,'entropy_coeff':.001})
try:
 torch.manual_seed(20261101)
 tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3.5-9B')
 cfg=OmegaConf.load(root/'verl/trainer/config/ppo_trainer.yaml').actor_rollout_ref.actor
 cfg.use_remove_padding=True;cfg.use_fused_kernels=False;cfg.ulysses_sequence_parallel_size=1
 model=AutoModelForImageTextToText.from_pretrained(root/'models/Qwen3.5-9B',dtype=torch.float32,attn_implementation='flash_attention_2').cuda()
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 actor=DataParallelPPOActor(cfg,model,None)
 prompts=[tokenizer.apply_chat_template([{'role':'user','content':'Your task is to clean a mug and put it on the table. '+('You see a sink, a mug, a table and a cabinet. '*n)+'Output only one action.'}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False) for n in [1,3,5,7]]
 responses=[tokenizer.encode('<action>go to sinkbasin 1</action>',add_special_tokens=False)+[tokenizer.eos_token_id] for _ in prompts]
 pw,rw=max(map(len,prompts)),max(map(len,responses));ids=torch.full((4,pw+rw),tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
 for i,(p,r) in enumerate(zip(prompts,responses)):
  ids[i,pw-len(p):pw]=torch.tensor(p,device='cuda');ids[i,pw:pw+len(r)]=torch.tensor(r,device='cuda');mask[i,pw-len(p):pw+len(r)]=1
 batch={'input_ids':ids,'responses':ids[:,pw:],'attention_mask':mask,'position_ids':(mask.cumsum(-1)-1).clamp_min(0)}
 model.eval()
 with torch.no_grad():
  _,old=actor._forward_micro_batch(batch,temperature=1.,calculate_entropy=False)
  singles=torch.cat([actor._forward_micro_batch({k:v[i:i+1] for k,v in batch.items()},temperature=1.,calculate_entropy=False)[1] for i in range(4)])
 valid=mask[:,pw:].bool();difference=(old-singles).abs()[valid]
 probe={'old_log_probs_finite':bool(torch.isfinite(old).all()),'single_log_probs_finite':bool(torch.isfinite(singles).all()),'packing_logprob_max_abs_difference':float(difference.max()),'packing_logprob_mean_abs_difference':float(difference.mean()),'prompt_lengths':list(map(len,prompts)),'response_tokens':rw,'optimizer_created':False,'parameter_updates':0}
 print('PACKING_FORWARD',json.dumps(probe),flush=True)
 model.train()
 for start in [0,2]:
  micro={k:v[start:start+2] for k,v in batch.items()}
  entropy,lp=actor._forward_micro_batch(micro,temperature=1.,calculate_entropy=True)
  response_mask=mask[start:start+2,pw:]
  loss,*_=compute_policy_loss(old[start:start+2].detach(),lp,torch.ones_like(lp),response_mask,cliprange=.2)
  entropy_loss=(entropy*response_mask).sum()/response_mask.sum()
  (loss-.001*entropy_loss).backward()
 torch.cuda.synchronize()
 bad=[name for name,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
 probe.update(gradients_finite=not bad,nonfinite_gradient_names=bad[:30],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
 (dest/'native_probe.json').write_text(json.dumps(probe,indent=2)+'\n')
 swanlab.log({k:v for k,v in probe.items() if isinstance(v,(int,float,bool))},step=0)
 print('NATIVE_PROBE',json.dumps(probe),flush=True)
 swanlab.finish(state='success')
except BaseException as error:
 (dest/'failure.txt').write_text(traceback.format_exc())
 swanlab.finish(state='crashed',error=type(error).__name__)
 raise
