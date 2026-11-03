import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from test_reskill_grpo import config,trainer,Tokenizer
from verl import DataProto
from verl.trainer.ppo.reskill_grpo import GRPOSkillRefinement
from verl.trainer.ppo.rewrite_quality import evaluate_rewritten_skills
from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld.projection import alfworld_projection

OBS='You are in a room. Your task is to: put the mug on the table.'
class Raw:
    instances=[]
    def __init__(self,*args,fixed_game_files=None,**kwargs):
        self.files=fixed_game_files or ['task-game','task-game'];self.get_admissible_commands=[['look'] for _ in self.files]
        self.closed=False;Raw.instances.append(self)
    def reset(self):return [OBS]*len(self.files),None,[{'extra.gamefile':f} for f in self.files]
    def step(self,actions):
        n=len(self.files)
        return ['You look at the room.']*n,None,np.ones(n)*10,np.ones(n,dtype=bool),[{'won':True,'extra.gamefile':f} for f in self.files]
    def close(self):self.closed=True
class Actor:
    world_size=4
    def __init__(self,tokenizer):self.tokenizer=tokenizer;self.requests=[]
    def generate_sequences(self,batch):
        self.requests.append(batch)
        rows=[self.tokenizer.encode('<action>look</action>')+[1] for _ in range(len(batch))]
        response=torch.tensor(rows);ids=torch.cat([batch.batch['input_ids'],response],-1)
        mask=torch.cat([batch.batch['attention_mask'],torch.ones_like(response)],-1)
        return DataProto.from_dict(tensors={'prompts':batch.batch['input_ids'],'responses':response,'input_ids':ids,'attention_mask':mask,'position_ids':(mask.cumsum(-1)-1).clamp_min(0)})
class NoThinkingTests(unittest.TestCase):
    def test_action_only_is_valid_in_training_and_quality_managers(self):
        for disabled in (False,True):
            cfg=config();cfg.env.disable_thinking=disabled
            env=AlfWorldEnvironmentManager(Raw(),alfworld_projection,cfg)
            obs,_=env.reset(None)
            self.assertEqual('MUST be enclosed' in obs['text'][0],not disabled)
            _,_,_,infos=env.step(['<action>look</action>']*2)
            self.assertEqual(bool(infos[0]['is_action_valid']),disabled)
        actions,valid=alfworld_projection(['look','<action>look</action>'],[['look']]*2,require_thinking=False)
        self.assertEqual(valid,[0,1])
    def test_native_quality_replay_preserves_task_seeds_and_valid_action_feedback(self):
        cfg=config();cfg.env.disable_thinking=True;cfg.algorithm.reskill_grpo.rewrite_quality.enable=True
        t=trainer(cfg);t.global_steps=5;c=GRPOSkillRefinement(t);actor=Actor(t.tokenizer)
        context={'group_size':2,'initial_gamefiles':['task-game']*2,'initial_anchors':[OBS]*2,
                 'pair_keys':['pair-0','pair-1'],'generation_meta_info':{'eos_token_id':1,'pad_token_id':0,'validate':False}}
        records=[{'group_index':0,'initial_skill':'old plan','refined_skill':'look around then put the mug',
                  'trajectories':[{'reward':10},{'reward':0}]}]
        with patch('agent_system.environments.env_package.alfworld.envs.AlfworldEnvs',Raw):
            self.assertEqual(evaluate_rewritten_skills(c,records,context,actor),1)
        result=records[0]['rewrite_quality']
        self.assertEqual(result['q'],1);self.assertEqual(result['p_new'],1.)
        self.assertTrue(Raw.instances[-1].closed)
        self.assertTrue(all(step['valid_action'] for traj in result['new_trajectories'] for step in traj))
        self.assertTrue(actor.requests[0].meta_info['rewrite_quality_execution'])
        self.assertIn('rewrite_request_seed',actor.requests[0].non_tensor_batch)
