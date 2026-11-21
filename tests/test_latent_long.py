from pathlib import Path
from types import SimpleNamespace
import unittest

from rev4.latent_long_plan import RECIPE,ARMS
from rev4.latent_plan import RECIPE as PILOT

try:
    from mmengine.config import Config
except ImportError:
    Config=None


class PlanTests(unittest.TestCase):
    def test_long_budget_does_not_mutate_pilot_or_model_design(self):
        self.assertEqual(PILOT['optimizer_updates'],1000)
        self.assertEqual(RECIPE['optimizer_updates'],10000)
        for key in ('seed','micro_batch_size','accumulation','learning_rate','weight_decay','clip_grad',
                    'frozen','trainable','data','loss','precision','plain_hidden','gated_hidden','primary_pair'):
            self.assertEqual(RECIPE[key],PILOT[key])

    @unittest.skipIf(Config is None,'Real MMEngine config required on HPC')
    def test_three_full_configs_differ_only_in_adapter_and_paths(self):
        from rev4.latent_trial import configuration
        root=Path(__file__).resolve().parents[1]
        a=SimpleNamespace(model_config=root/'docs/handoff/inference_config.py',calibration_prepared=Path('prepared'),
                          output=Path('long'),checkpoint=Path('b0.pth'),latent_recipe=RECIPE,
                          latent_audit_hook='LatentLongAuditHook',latent_extra_imports=['rev4.latent_long_plugins'])
        normalized=[]
        for arm in ARMS:
            cfg=configuration(a,arm)
            self.assertEqual(cfg.train_cfg.max_iters,20000)
            self.assertEqual(cfg.param_scheduler[0]['end'],200)
            self.assertEqual(cfg.param_scheduler[1]['end'],20000)
            self.assertEqual(cfg.default_hooks.checkpoint.interval,5000)
            self.assertEqual(cfg.default_hooks.checkpoint.max_keep_ckpts,4)
            self.assertIsNone(cfg.load_from)
            self.assertFalse(cfg.resume)
            self.assertIs(cfg.model.backbone['freeze'],True)
            self.assertEqual(cfg.custom_hooks,[dict(type='LatentLongAuditHook')])
            value=cfg.to_dict();value['model'].pop('adapter_kind');value.pop('work_dir')
            normalized.append(value)
        self.assertEqual(normalized[0],normalized[1]);self.assertEqual(normalized[1],normalized[2])


if __name__=='__main__':unittest.main()
