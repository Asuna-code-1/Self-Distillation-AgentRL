"""Finish native dependencies, then build the exact FlashAttention pin for H200."""
import os,signal,subprocess,time
from pathlib import Path
import psutil
ROOT=Path('/workspace/fudan/donghaosen/Self-Distillation-AgentRL')
LOG=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training')
parent_pid=2677694
while True:
 status=LOG/'bootstrap-v2.exit'
 if status.exists():
  break
 try:parent=psutil.Process(parent_pid)
 except psutil.NoSuchProcess:break
 switched=False
 for child in parent.children(recursive=True):
  try:
   cmd=child.cmdline()
   if '-m' in cmd and 'pip' in cmd and 'flash-attn==2.8.3' in cmd:
    for sub in reversed(child.children(recursive=True)):
     try:sub.send_signal(signal.SIGTERM)
     except psutil.Error:pass
    child.send_signal(signal.SIGTERM);switched=True
  except psutil.Error:pass
 if switched:
  while not status.exists():time.sleep(.2)
  break
 time.sleep(.5)
with (LOG/'extensions-h200.log').open('w') as output:
 result=subprocess.run(['bash',str(ROOT/'launchers/local_train/install_extensions_h200.sh')],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
(LOG/'extensions-h200.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
