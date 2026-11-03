import json
from pathlib import Path
from agent_system.environments.env_package.alfworld.envs import AlfworldWorker,load_config_file,get_environment
root=Path(__file__).resolve().parents[2]
cfg=load_config_file(str(root/'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'))
def base():return get_environment(cfg['env']['type'])(cfg,train_eval='train')
original=AlfworldWorker(cfg,123,base())
try:
    original.reset()
    obs,info=original.reset()
    game=info['extra.gamefile'][0]
    clone=AlfworldWorker(cfg,123,base(),fixed_game_file=game)
    try:
        new_obs,new_info=clone.reset()
        assert new_info['extra.gamefile']==info['extra.gamefile']
        assert new_obs==obs,(new_obs,obs)
        clone.step('look')
        assert original._active_gamefile==game
        original.reset()
        assert original._active_gamefile!=game
        result={'same_game':True,'same_initial_observation':True,'original_iterator_unaffected':True,'game':game}
        target=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/fixed-game-replay.json')
        target.write_text(json.dumps(result,indent=2)+'\n')
        print('FIXED_GAME_REPLAY_OK',flush=True)
    finally:clone.env.close()
finally:original.env.close()
