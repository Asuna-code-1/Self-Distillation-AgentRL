"""Real packed Qwen3.5 backward preflight; no optimizer/training data changes."""
import json,threading,time
from pathlib import Path
import torch,pynvml
from transformers import AutoModelForImageTextToText
from omegaconf import OmegaConf
from verl.workers.actor.dp_actor import DataParallelPPOActor
root=Path(__file__).resolve().parents[2]
cfg=OmegaConf.load(root/'verl/trainer/config/ppo_trainer.yaml').actor_rollout_ref.actor
cfg.use_remove_padding=True;cfg.use_fused_kernels=False;cfg.ulysses_sequence_parallel_size=1
pynvml.nvmlInit();h=pynvml.nvmlDeviceGetHandleByIndex(0);peaks=[0.];stop=threading.Event()
def sample():
 while not stop.wait(.1):peaks[0]=max(peaks[0],pynvml.nvmlDeviceGetMemoryInfo(h).used/2**30)
threading.Thread(target=sample,daemon=True).start()
model=AutoModelForImageTextToText.from_pretrained(root/'models/Qwen3.5-4B',dtype=torch.float32,attn_implementation='flash_attention_2').cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
actor=DataParallelPPOActor(cfg,model,None)
count,length,response=14,2048,512
torch.manual_seed(123)
ids=torch.randint(100,10000,(count,length),device='cuda')
batch={'input_ids':ids,'responses':ids[:,-response:],'attention_mask':torch.ones_like(ids),
       'position_ids':torch.arange(length,device='cuda')[None,:].expand(count,-1)}
torch.cuda.reset_peak_memory_stats()
entropy,lp=actor._forward_micro_batch(batch,temperature=1.,calculate_entropy=True)
loss=-lp.float().mean()-float(cfg.entropy_coeff)*entropy.float().mean();assert torch.isfinite(loss)
loss.backward();torch.cuda.synchronize()
finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
assert finite
stop.set();result={'tokens':count*length,'loss_finite':True,'gradients_finite':finite,'sampled_peak_gib':peaks[0],
                   'torch_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'full_model_single_gpu_preflight':True,'entropy_coeff':float(cfg.entropy_coeff)}
Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/native-backward-entropy-4b-20261031.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert peaks[0]<130
