"""Regression checks for the frozen calibration grid and validation cohort."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from lowres.common import write_jsonl
from rev4 import calibration
from rev4.g4_review import diagnostic_summary


class CalibrationTests(TestCase):
    def test_same_grid_uses_micro_counts_and_native_tie_break(self):
        tasks = [{'condition': condition} for condition in ('clean', 'lr2')]
        counts = [[[1, 2, 3, 1] for _ in calibration.GRID] for _ in tasks]
        grid = calibration.summarize_grid(counts, tasks)
        self.assertEqual(len(grid), 15)
        self.assertEqual(grid[0]['scores']['lr2']['hmean'], .4)
        self.assertIs(calibration.choose_threshold(list(reversed(grid))), grid[0])
        counts[0][5] = [2, 2, 2, 2]
        updated = calibration.summarize_grid(counts, tasks)
        self.assertIs(calibration.choose_threshold(updated), updated[5])

    def test_cohort_rejects_an_extra_task_and_s2(self):
        a = SimpleNamespace(original_remaining_ids=['kept', 'pilot2'], s1_ids=['pilot1'], s2_ids=['pilot2'])
        with TemporaryDirectory() as temporary, patch.object(calibration, 'VAL_SOURCES', 1):
            folder = Path(temporary)
            rows = folder/'validation_remaining.jsonl'
            tasks = folder/'validation_tasks.jsonl'
            clean, lr2 = (dict(source_image_id='kept', condition=c) for c in ('clean', 'lr2'))
            write_jsonl(rows, [dict(source_image_id='kept')])
            write_jsonl(tasks, [clean, lr2])
            self.assertEqual(calibration.validation_cohort(a, folder)[1], [clean, lr2])
            write_jsonl(tasks, [clean, lr2, lr2])
            with self.assertRaisesRegex(ValueError, 'Expected paired'):
                calibration.validation_cohort(a, folder)
            write_jsonl(rows, [dict(source_image_id='pilot2')])
            with self.assertRaisesRegex(ValueError, 'frozen'):
                calibration.validation_cohort(a, folder)

    def test_g4_denominators_and_pending_review(self):
        eligible = dict(native_height=6, kernel_pixels=4, primary='weak_response', sensitivity='weak_response',
                        coverage_clean=.55, network_input_height=18)
        invalid = dict(eligible, kernel_pixels=3, primary='invalid_kernel', coverage_clean=None)
        summary = dict(conditions=dict(lr2=dict(failures=[eligible, invalid], primary_failure_counts={},
                                                 native_height_bins={})), b_up_lr2_gain_pp=.8,
                       b_up_lr2_gain_95ci_pp=[.3, 1.3], b_up_lr2_recovery_ratio=.08)
        result = diagnostic_summary(summary)
        self.assertEqual(result['review_status'], 'PENDING_HUMAN_REVIEW')
        self.assertEqual(result['invalid_lr2_kernels'], 1)
        self.assertEqual(result['strata'][2]['valid_kernel_misses'], 1)
        self.assertEqual(result['strata'][2]['clean_coverage_ge_050'], 1)
        self.assertEqual(result['strata'][3]['valid_kernel_misses'], 0)
