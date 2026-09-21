import json
import tempfile
import unittest
from pathlib import Path
from nimble.training.tune_mixed_schema import completed_trial_history


class ReducedTuningTests(unittest.TestCase):
    def test_requires_complete_matching_checkpoint_before_reusing_metrics(self):
        plan={'epochs':[1,2],'model':'model','revision':'revision','batch_size':8,
              'gradient_accumulation':1,'inner_training_fingerprint':'training'}
        config={'learning_rate':2e-5,'lora_rank':16}
        contract={**config,'model':'model','revision':'revision','batch_size':8,
                  'gradient_accumulation':1,'max_steps':909,'warmup_steps':91,
                  'data_audit':{'training_fingerprint':'training'}}
        history=[{'epoch':1},{'epoch':2}]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'checkpoint-606';checkpoint.mkdir()
            (root/'epoch_metrics.json').write_text(json.dumps(history))
            (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':606,'epoch':2.0}))
            (checkpoint/'schema_config.json').write_text(json.dumps(contract))
            with self.assertRaises(ValueError):completed_trial_history(root,config,plan,303)
            (checkpoint/'adapter_model.safetensors').touch()
            self.assertEqual(completed_trial_history(root,config,plan,303),history)
            with self.assertRaises(ValueError):
                completed_trial_history(root,{**config,'learning_rate':5e-5},plan,303)
            with self.assertRaises(ValueError):
                completed_trial_history(root,config,{**plan,'inner_training_fingerprint':'other'},303)
            (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':605,'epoch':1.99}))
            with self.assertRaises(ValueError):completed_trial_history(root,config,plan,303)


if __name__=='__main__':unittest.main()
