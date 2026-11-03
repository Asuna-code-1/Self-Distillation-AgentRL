"""Check the completion gate and the prescribed stop sequence without GPU jobs."""
import importlib.util,json,unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handoff',ROOT/'launchers/local_train/stop_at_step_then_restart.py')
handoff=importlib.util.module_from_spec(spec);spec.loader.exec_module(handoff)

class HandoffTests(unittest.TestCase):
    def fixture(self):
        directory=ROOT/'local_assets/tests'/uuid4().hex;directory.mkdir(parents=True)
        log=directory/'console.log';log.write_text('step:99 - val/success_rate:0.8\n')
        record={'name':'fixture','console_log':str(log),'validation_size':2,'attempt_directory':str(directory),'session':'fixture-session'}
        base=directory/'checkpoints/reskill_grpo/alfworld/fixture'
        checkpoint=base/'global_step_100';(checkpoint/'actor').mkdir(parents=True)
        files=[checkpoint/'data.pt']+[checkpoint/'actor'/f'{kind}_world_size_4_rank_{rank}.pt' for kind in ['model','optim','extra_state'] for rank in range(4)]
        for p in files:p.write_bytes(b'fixture')
        validation=base/'validation_skills/step-100.jsonl';validation.parent.mkdir()
        validation.write_text(''.join(json.dumps({'step':100})+'\n' for _ in range(2)))
        return directory,record,files,validation

    def test_waits_for_completed_step_log_after_all_artifacts_exist(self):
        directory,record,files,validation=self.fixture()
        self.assertFalse(handoff.checkpoint_ready(record,100,directory))
        Path(record['console_log']).write_text('step:100 - val/success_rate:0.9\n')
        self.assertTrue(handoff.checkpoint_ready(record,100,directory))

    def test_incomplete_checkpoint_or_validation_cannot_trigger_stop(self):
        directory,record,files,validation=self.fixture()
        Path(record['console_log']).write_text('step:100 - val/success_rate:0.9\n')
        files[-1].write_bytes(b'')
        self.assertFalse(handoff.checkpoint_ready(record,100,directory))
        files[-1].write_bytes(b'fixture')
        validation.write_text('{"step":100}\n')
        self.assertFalse(handoff.checkpoint_ready(record,100,directory))
        validation.write_text('{"step":100}\n{"step":99}\n')
        self.assertFalse(handoff.checkpoint_ready(record,100,directory))

    def test_wrong_step_number_does_not_match_target(self):
        directory,record,_,_=self.fixture()
        Path(record['console_log']).write_text('step:1000 - val/success_rate:0.9\n')
        self.assertFalse(handoff.checkpoint_ready(record,100,directory))

    def test_uses_ctrl_z_then_job_kill_before_releasing_old_group(self):
        directory,record,_,_=self.fixture()
        with patch.object(handoff.subprocess,'run') as run,patch.object(handoff.time,'sleep'),patch.object(handoff,'live_group',return_value=[]):
            handoff.stop_run(record,12345)
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0],['tmux','send-keys','-t','fixture-session','C-z'])
        self.assertEqual(commands[1][-1],'kill %1')
        self.assertEqual(commands[3][-1],'kill -CONT %1')
        self.assertEqual((directory/'launcher.exit').read_text(),'143\n')

if __name__=='__main__':unittest.main()
