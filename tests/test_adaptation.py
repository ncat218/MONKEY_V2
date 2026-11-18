"""Control fairness, evaluator equivalence, paired arithmetic and launch guards."""
import copy
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shapely.geometry import box, mapping

from lowres.common import read_json, save_json
from lowres.evaluate_shared import evaluate_image
from rev4 import adaptation_jobs, bclean
from rev4.paired_eda_core import (assert_transition_counts, fp_transitions, gt_attributes,
                                  instance_evidence, trace_matching, transition)
from rev4.paired_eda_report import summarize
from rev4.training_data import balanced_indices


class ConfigNode(dict):
    def __init__(self, **items):
        super().__init__({key: self.convert(value) for key, value in items.items()})

    @classmethod
    def convert(cls, value):
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, list):
            return [cls.convert(item) for item in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self.convert(value)


def config_fixture():
    return ConfigNode(load_from='b0.pth', resume=False, train_cfg=dict(max_iters=20000),
                      optim_wrapper=dict(optimizer=dict(lr=2e-5), accumulative_counts=2),
                      custom_imports=dict(imports=['rev4.b1_plugins']),
                      train_dataloader=dict(sampler=dict(type='Rev4BalancedLR2Sampler'),
                          dataset=dict(ann_file='old.json', pipeline=[dict(type='OriginalAugmentations'),
                              dict(type='PackTextDetInputs', meta_keys=('condition', 'sample_id'))])))

POLICY = dict(min_prediction_roi_fraction=.5, ignore_prediction_overlap=.5, match_iou=.5, word_coverage=.5)


def rectangle(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def fixture():
    gt = dict(ann_id='word', geometry=mapping(box(10, 10, 30, 20)), original_height=10)
    row = dict(clean_size=[100, 100], roi_clean=mapping(box(0, 0, 100, 100)),
               gt_clean=dict(valid=[gt], ignored=[dict(geometry=mapping(box(70, 70, 90, 90)))]))
    kernel = {'word': dict(kernel_pixels=10, coverage=.7)}
    return row, kernel


class AdaptationTests(unittest.TestCase):
    def test_clean_control_keeps_exact_source_draws_and_full_clean_geometry(self):
        clean = [dict(condition='clean', sample_id=f'clean::{i}', img_path=f'{i}.jpg',
                      height=100, width=100, instances=[dict(polygon=rectangle(10, 10, 30, 20))]) for i in range(3)]
        low = [dict(item, condition='lr2', sample_id=f'lr2::{i}', height=50, width=50,
                    instances=[dict(polygon=rectangle(5, 5, 15, 10))]) for i, item in enumerate(clean)]
        original = dict(data_list=clean+low)
        snapshot = copy.deepcopy(original)
        with patch.object(bclean, 'TRAIN_SOURCES', 3):
            result = bclean.clean_groups(original)
        self.assertEqual(original, snapshot)
        self.assertTrue(all(item['condition'] == 'clean' and item['height'] == 100 for item in result['data_list']))
        self.assertEqual([item['clean_draw_slot'] for item in result['data_list']], [0, 0, 0, 1, 1, 1])
        stream = list(itertools.islice(balanced_indices(3, 42), 24))
        for index in stream:
            self.assertEqual(original['data_list'][index]['img_path'], result['data_list'][index]['img_path'])
            self.assertEqual(result['data_list'][index]['instances'], clean[index % 3]['instances'])

    def test_mismatched_training_source_groups_stop(self):
        data = [dict(condition='clean', sample_id='clean::a', img_path='a'),
                dict(condition='lr2', sample_id='lr2::b', img_path='b')]
        with patch.object(bclean, 'TRAIN_SOURCES', 1), self.assertRaisesRegex(ValueError, 'do not correspond'):
            bclean.clean_groups(dict(data_list=data))

    def test_control_preserves_budget_optimizer_and_augmentation(self):
        base = config_fixture()
        base.load_from = 'b0.pth'
        base.train_dataloader.sampler = dict(type='Rev4BalancedLR2Sampler')
        base.train_dataloader.dataset.ann_file = 'old.json'
        base.train_dataloader.dataset.pipeline[-1]['meta_keys'] = ('condition', 'sample_id')
        a = SimpleNamespace(checkpoint=Path('b0.pth'), output=Path('new'), calibration_prepared=Path('old'))
        with patch.object(bclean.b1_training, 'configuration', return_value=copy.deepcopy(base)):
            control = bclean.configuration(a, Path('clean.json'))
        self.assertEqual(control.train_cfg, base.train_cfg)
        self.assertEqual(control.optim_wrapper, base.optim_wrapper)
        self.assertEqual(control.train_dataloader.dataset.pipeline[:-1], base.train_dataloader.dataset.pipeline[:-1])
        self.assertEqual(control.load_from, 'b0.pth')
        self.assertFalse(control.resume)
        self.assertEqual(control.train_dataloader.dataset.ann_file, 'clean.json')
        self.assertEqual(control.train_dataloader.sampler.type, 'CleanControlSampler')

    def test_exact_match_trace_preserves_invalid_fp_and_ignored_regions(self):
        row, _ = fixture()
        prediction = dict(polygons=[rectangle(10, 10, 30, 20), rectangle(72, 72, 80, 80),
                                    rectangle(120, 120, 130, 130), [1, 1], rectangle(40, 40, 50, 50)],
                          scores=[.8]*5)
        official, assigned, _, fp = trace_matching(prediction, row, POLICY)
        self.assertEqual(official, evaluate_image(row['gt_clean'], row['roi_clean'], prediction, POLICY))
        self.assertEqual(set(assigned), {'word'})
        self.assertEqual(set(fp), {3, 4})
        self.assertEqual(official['counts'][:3], [1, 1, 3])

    def test_one_to_one_ambiguous_assignment_and_missing_candidate(self):
        row, kernel = fixture()
        row['gt_clean']['valid'].append(dict(row['gt_clean']['valid'][0], ann_id='second'))
        kernel['second'] = kernel['word']
        prediction = dict(polygons=[rectangle(10, 10, 30, 20)], scores=[.8])
        evidence = instance_evidence(prediction, prediction, row, POLICY, kernel, .4)
        self.assertEqual(sum(v['tp'] for v in evidence['instances'].values()), 1)
        self.assertEqual(sum(v['diagnosis'] == 'assignment_ambiguous' for v in evidence['instances'].values()), 1)
        empty = dict(polygons=[], scores=[])
        missing = instance_evidence(empty, empty, row, POLICY, kernel, .4)['instances']['word']
        self.assertIsNone(missing['candidate_score'])
        self.assertIsNone(missing['candidate_score_margin'])
        self.assertEqual(missing['diagnosis'], 'no_candidate')

    def test_score_filtering_and_fp_net_are_separate_from_gt_matching(self):
        row, kernel = fixture()
        empty = dict(polygons=[], scores=[])
        candidate = dict(polygons=[rectangle(10, 10, 30, 20)], scores=[.35])
        self.assertEqual(instance_evidence(empty, candidate, row, POLICY, kernel, .4)['instances']['word']['diagnosis'], 'score_filtering')
        old = instance_evidence(dict(polygons=[rectangle(40, 40, 50, 50)], scores=[.6]), empty, row, POLICY, kernel, .4)
        new = instance_evidence(dict(polygons=[rectangle(40, 40, 50, 50), [1, 1]], scores=[.6, .6]), empty, row, POLICY, kernel, .4)
        self.assertEqual([v['kind'] for v in fp_transitions(old, new, row, POLICY)], ['persistent_fp', 'newer_only_fp'])
        assert_transition_counts({'rescued': 2, 'regressed': 1, 'persistent_tp': 3, 'persistent_fn': 4}, 4, 5, 10)
        with self.assertRaises(ValueError):
            assert_transition_counts({'rescued': 2}, 1, 2, 2)
        self.assertEqual(transition(True, False), 'regressed')

    def test_report_recomputes_micro_and_all_three_model_transitions(self):
        row, kernel = fixture()
        full = dict(polygons=[rectangle(10, 10, 30, 20)], scores=[.8])
        empty = dict(polygons=[], scores=[])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_json(root/'policy.json', POLICY)
            rows = {'a': row, 'b': row}
            tasks = [dict(source_image_id=s, condition=c, sample_id=f'{c}::{s}') for c in ('clean', 'lr2') for s in rows]
            names = ['B0', 'B-Clean', 'B1']
            for name in names:
                for task in tasks:
                    source, condition = task['source_image_id'], task['condition']
                    prediction = empty if source == 'b' and name == 'B0' else full
                    result = instance_evidence(prediction, prediction, row, POLICY, kernel, .4)
                    save_json(root/'models'/name/condition/f'{source}.json', dict(
                        **task, modes={mode: '.3/.4' for mode in ('native', 'common', 'calibrated')},
                        results={'.3/.4': result}, gt_attributes={'word': gt_attributes(row['gt_clean']['valid'][0], row['clean_size'])},
                        kernel_evidence=kernel, polygon_probability_statistics={'word': dict(mean=.7, q10=.5, median=.7, q90=.9)},
                        dimensions=dict(native_size=[100, 100], resized_hw=[100, 100], padded_input=[1, 3, 128, 128],
                                        pmap=[1, 128, 128], backbone=[[1, 64, 32, 32]], neck=[1, 256, 32, 32])))
            a = SimpleNamespace(output=root, mode='B0_BClean_B1', evaluation_protocol=root/'policy.json')
            cases, output = summarize(a, [dict(name=name) for name in names], rows, tasks)
            summary = read_json(output/'eda_summary.json')
            self.assertAlmostEqual(summary['scores']['B0']['common']['lr2']['hmean'], 2/3)
            self.assertEqual(summary['scores']['B1']['common']['lr2']['hmean'], 1.)
            comparison = next(c for c in summary['comparisons'] if c['older'] == 'B0' and c['newer'] == 'B1' and c['mode'] == 'common' and c['condition'] == 'lr2')
            self.assertEqual(comparison['transitions']['rescued'], 1)
            self.assertAlmostEqual(comparison['hmean_difference_pp'], 100/3)
            self.assertEqual(comparison['paired_source_bootstrap']['repetitions'], 2000)
            self.assertTrue(cases)
            self.assertTrue((output/'paired_transitions.csv').is_file())

    def test_launcher_is_lightweight_and_dependencies_are_afterok(self):
        with patch.object(adaptation_jobs, 'source_identity', return_value={'fixture': 'hash'}):
            a = adaptation_jobs.light_inputs('eda', True)
        command = adaptation_jobs.sbatch_args(a, ['123', '456'])
        self.assertIn('--dependency=afterok:123:456', command)
        worker = shlex.split(command[-1].split('; exec ', 1)[1])
        self.assertEqual(worker[-2:], ['--worker', '--include-bclean'])
        self.assertIn('--mem=40G', command)
        with self.assertRaises(ValueError):
            adaptation_jobs.sbatch_args(a, ['123;bad'])
        for launcher in ('submit_bclean.py', 'submit_paired_eda.py'):
            probe = subprocess.run([sys.executable, '-c',
                f'import runpy,sys; sys.argv=["{launcher}","--plan-only"]; runpy.run_path("{launcher}",run_name="__main__"); '
                'assert "torch" not in sys.modules and "numpy" not in sys.modules'], capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            json.loads(probe.stdout)


if __name__ == '__main__':
    unittest.main()
