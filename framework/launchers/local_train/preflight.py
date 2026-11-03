"""Actual Qwen3.5 2B packed-vs-independent numerical check on GPU 0."""
import json,os
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForImageTextToText,AutoTokenizer
from verl.models.transformers.packed_hybrid import packed_hybrid_kwargs
ROOT=Path('/workspace/fudan/donghaosen/Self-Distillation-AgentRL')
model_path=ROOT/'models/Qwen3.5-2B'
config=AutoConfig.from_pretrained(model_path,local_files_only=True)
try:
 from transformers import AutoModelForVision2Seq
except ImportError:
 AutoModelForVision2Seq=AutoModelForImageTextToText
print('Native model-class mapping recognizes config:',type(config) in AutoModelForVision2Seq._model_mapping.keys(),flush=True)
assert type(config) in AutoModelForVision2Seq._model_mapping.keys()
model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='flash_attention_2').eval()
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
texts=['Your task is to put a clean mug on a table.','You are in a kitchen. Find a cold apple.']
inputs=[tokenizer.encode(text,add_special_tokens=False,return_tensors='pt').to('cuda:0') for text in texts]
references=[]
with torch.no_grad():
 for ids in inputs:
  positions=torch.arange(ids.shape[1],device=ids.device).unsqueeze(0)
  references.append(model(input_ids=ids,attention_mask=None,position_ids=positions,use_cache=False).logits)
 packed_ids=torch.cat(inputs,dim=1)
 positions=torch.cat([torch.arange(ids.shape[1],device=ids.device) for ids in inputs]).unsqueeze(0)
 kwargs=packed_hybrid_kwargs(config,positions)
 packed=model(input_ids=packed_ids,attention_mask=None,position_ids=positions,use_cache=False,**kwargs).logits
 target=torch.cat(references,dim=1)
 difference=(packed.float()-target.float()).abs()
 max_error=float(difference.max());mean_error=float(difference.mean())
 torch.testing.assert_close(packed,target,atol=.15,rtol=.02)
 assert mean_error<.025,(max_error,mean_error)
 report={'native_auto_model_mapping':True,'independent_vs_packed_passed':True,'max_abs_error':max_error,'mean_abs_error':mean_error,'input_lengths':[ids.shape[1] for ids in inputs],'packed_boundaries':kwargs['cu_seq_lens_q'].tolist(),'conv_sequence_ids_passed':True,'gdn_boundaries_passed':True}
 print(json.dumps(report,indent=2),flush=True)
 Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/packed-model-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
