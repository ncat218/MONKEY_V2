"""Equal-budget setup, Slurm plan and paired-summary regression checks."""
import copy
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from lowres.common import read_json, save_json
from rev4 import feature_pilot
import submit_feature_pilot


class ConfigNode(dict):
    """Minimal recursive config fixture; exercises production config mutations."""
    def __init__(self, **items):
        super().__init__({k: self.convert(v) for k, v in items.items()})

    @classmethod
    def convert(cls, value):
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, list):
            return [cls.convert(v) for v in value]
        return value

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = self.convert(value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = self.convert(value)


def config_fixture():
    return ConfigNode(load_from='original', resume=False,
                      train_cfg=dict(max_iters=20_000),
                      default_hooks=dict(checkpoint=dict(interval=5000, max_keep_ckpts=3)),
                      optim_wrapper=dict(optimizer=dict(lr=2e-5)),
                      custom_imports=dict(imports=['original']),
                      model=dict(det_head=dict(type='DBHead')),
                      train_dataloader=dict(dataset=dict(pipeline=[
                          dict(type=name) for name in ('Rev4LoadCleanOrLR2', 'LoadOCRAnnotations',
                                                       'TorchVisionWrapper', 'ImgAugWrapper', 'RandomCrop', 'Resize')]
                          +[dict(type='Pad', size=(640, 640)), dict(type='PackTextDetInputs')])))


class FeaturePilotTests(TestCase):
    def test_arms_have_equal_budget_init_and_transform_pipeline(self):
        a = SimpleNamespace(output=Path('pilot'), calibration_prepared=Path('existing'), b1_checkpoint=Path('b1.pth'))
        with patch.object(feature_pilot.b1_training, 'configuration', side_effect=lambda *args: config_fixture()):
            control, method = [feature_pilot.configuration(a, arm) for arm in feature_pilot.ARMS]
        self.assertEqual(control.load_from, method.load_from)
        self.assertEqual(control.load_from, 'b1.pth')
        self.assertEqual(control.train_cfg.max_iters, 2000)
        self.assertEqual(control.train_cfg, method.train_cfg)
        self.assertEqual(control.optim_wrapper, method.optim_wrapper)
        self.assertEqual(control.param_scheduler, method.param_scheduler)
        self.assertEqual(control.train_dataloader, method.train_dataloader)
        self.assertEqual(control.custom_hooks, method.custom_hooks)
        self.assertEqual(control.model.det_head.type, 'DBHead')
        self.assertEqual(method.model.det_head.type, 'FeaturePilotHead')
        self.assertEqual(control.default_hooks.checkpoint.interval, 2000)

    def test_launcher_worker_uses_existing_stack_and_has_no_test_phase(self):
        a = SimpleNamespace(s1=Path('/existing/pillow stack'), cuda=Path('/existing/cuda'),
                            python=Path('/existing/venv/bin/python'))
        command = submit_feature_pilot.sbatch_args(a)
        self.assertIn('--gres=gpu:2g.35gb:1', command)
        self.assertIn('--time=06:00:00', command)
        worker = shlex.split(command[-1].split('; exec ', 1)[1])
        self.assertEqual(worker[0], str(a.python))
        self.assertEqual(worker[-1], '--worker')
        self.assertEqual(len(feature_pilot.PAIRS), 2)
        self.assertEqual(feature_pilot.PAIRS[1], (.3, .4))

    def test_failed_preflight_does_not_start_training(self):
        import subprocess

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = SimpleNamespace(python=Path('python'), output=root, pilot_identity={'fixture': 1})
            with (patch.object(submit_feature_pilot, 'RUN', root),
                  patch.object(submit_feature_pilot, 'lock_identity'),
                  patch.object(submit_feature_pilot.subprocess, 'run'),
                  patch.object(submit_feature_pilot, 'child', side_effect=subprocess.CalledProcessError(1, 'probe')) as child):
                # Windows lacks fcntl; use a small mock for the orchestration-only test.
                import sys
                fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
                with patch.dict(sys.modules, {'fcntl': fcntl}):
                    with self.assertRaises(subprocess.CalledProcessError):
                        submit_feature_pilot.worker(a)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(child.call_args.args[1:], ('--phase', 'probe'))

    def test_summary_pairs_sources_and_recomputes_micro_hmean(self):
        # The macro average differs from micro; method/control differ only for LR2.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {'a': {}, 'b': {}}
            tasks = [dict(source_image_id=s, condition=c) for c in ('clean', 'lr2') for s in rows]
            identity = {'fixture': True}
            reference = root/'reference.json'
            baseline = [[1, 1, 1, 0], [0, 9, 0, 0], [1, 1, 1, 0], [0, 9, 0, 0]]
            save_json(reference, dict(identity=dict(grid=[list(p) for p in feature_pilot.PAIRS]),
                                      counts=[[counts, counts] for counts in baseline]))
            for arm in feature_pilot.ARMS:
                arm_tasks = []
                for task, counts in zip(tasks, baseline):
                    counts = copy.copy(counts)
                    if arm == 'feature_kd' and task['condition'] == 'lr2' and task['source_image_id'] == 'b':
                        counts = [1, 9, 1, 0]
                    arm_tasks.append(dict(task, sample_id=f'{task["condition"]}::{task["source_image_id"]}',
                                          counts=[counts, counts], bins=[[[1, 0]]*4]*2, native_latency_ms=1.))
                save_json(root/arm/'validation.json', dict(identity=identity, tasks=arm_tasks))
                save_json(root/arm/'trained.json', dict(active_microbatches=2000))
            a = SimpleNamespace(output=root, calibration_prepared=root, calibration_counts=reference, pilot_identity=identity)
            with patch.object(feature_pilot.calibration, 'validation_cohort', return_value=(rows, tasks)):
                feature_pilot.summarize(a)
            result = read_json(root/'results/pilot_summary.json')
            self.assertAlmostEqual(result['scores']['B1']['lr2'][1]['hmean'], 2/11)
            self.assertAlmostEqual(result['scores']['feature_kd']['lr2'][1]['hmean'], 4/12)
            self.assertEqual(result['differences']['feature_kd_minus_continued']['clean'][1]['hmean_pp'], 0.)
            self.assertGreater(result['differences']['feature_kd_minus_continued']['lr2'][1]['hmean_pp'], 0.)
            self.assertEqual(result['bootstrap_repetitions'], 2000)


if __name__ == '__main__':
    import unittest
    unittest.main()
