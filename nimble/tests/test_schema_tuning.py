import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nimble.training.tune_schema import choose_candidate


class TuningTests(unittest.TestCase):
    def test_selection_uses_inner_nll_and_rejects_nonfinite(self):
        a = {'epoch': 3, 'learning_rate': 1e-5, 'summary': {'all': {'nll': .3, 'brier': .2}}, 'outer_accuracy': 1.0}
        b = {'epoch': 2, 'learning_rate': 2e-5, 'summary': {'all': {'nll': .2, 'brier': .3}}, 'outer_accuracy': 0.0}
        self.assertEqual(choose_candidate([a, b]), b)
        self.assertEqual(choose_candidate([{**b,'lora_rank':32},{**b,'lora_rank':16}])['lora_rank'],16)
        with self.assertRaises(ValueError):
            choose_candidate([{**a, 'summary': {'all': {'nll': float('nan')}}}])

    def test_epoch_callback_restores_training_and_stops_only_at_boundary(self):
        from nimble.training.schema_train import EpochValidation
        model = SimpleNamespace(training=True)
        model.train = lambda flag: setattr(model, 'training', flag)
        def fake_evaluate(*args):
            model.training = False
            return {'summary': {'all': {'nll': .2}}}
        with tempfile.TemporaryDirectory() as directory, patch('nimble.training.schema_train.evaluate', fake_evaluate):
            callback = EpochValidation([], None, Path(directory), stop_after_epochs=2)
            control = SimpleNamespace(should_training_stop=False)
            for epoch, step in [(1., 100), (1., 100), (1.5, 150), (2., 200)]:
                callback.on_epoch_end(None, SimpleNamespace(epoch=epoch, global_step=step), control, model=model)
            self.assertTrue(model.training)
            self.assertTrue(control.should_training_stop)
            self.assertEqual([r['epoch'] for r in callback.history], [1, 2])
            restored = EpochValidation([], None, Path(directory))
            self.assertEqual(len(restored.history), 2)


if __name__ == '__main__':
    unittest.main()
