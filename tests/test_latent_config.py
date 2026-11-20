"""Exercise real MMEngine configs, including keys that collide with dict methods."""
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    from mmengine.config import Config
except ImportError:
    Config=None


@unittest.skipIf(Config is None,'Requires historical MMEngine; mandatory on HPC')
class LatentConfigTests(unittest.TestCase):
    def test_training_and_inference_configs_preserve_frozen_model_contract(self):
        from rev4.latent_trial import configuration,inference_configuration
        config=Path(__file__).resolve().parents[1]/'docs/handoff/inference_config.py'
        a=SimpleNamespace(model_config=config,calibration_prepared=Path('prepared'),
                          output=Path('trial'),checkpoint=Path('b0.pth'))
        reference=Config.fromfile(str(config))
        for arm in ('head','plain','gated'):
            train=configuration(a,arm)
            inference=inference_configuration(a,arm)
            for cfg in (train,inference):
                self.assertIs(cfg.model.backbone['freeze'],True)
                self.assertTrue(callable(cfg.model.backbone.freeze))
                self.assertEqual(cfg.model.type,'LatentPilotDBNet')
                self.assertEqual(cfg.model.adapter_kind,arm)
                self.assertIn('rev4.latent_plugins',cfg.custom_imports['imports'])
                self.assertEqual(cfg.model.det_head,reference.model.det_head)
                self.assertEqual(cfg.model.neck,reference.model.neck)
            self.assertIsNone(train.load_from)
            self.assertEqual(train.train_cfg.max_iters,2000)
            self.assertEqual(train.optim_wrapper.accumulative_counts,2)
            self.assertEqual(train.optim_wrapper.optimizer.lr,2e-5)
            self.assertEqual(train.custom_hooks,[dict(type='LatentAuditHook')])


if __name__=='__main__':unittest.main()
