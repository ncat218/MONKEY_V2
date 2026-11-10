"""Regression checks for calibrated matching, residual taxonomy and paired counts."""
from unittest import TestCase
from unittest.mock import patch

from rev4 import b1_diagnostics as diagnostics


class B1DiagnosticTests(TestCase):
    def test_mask_response_precedes_polygon_diagnosis(self):
        self.assertEqual(diagnostics.reason(dict(kernel_pixels=3, coverage=None), [], [.8], .5), 'invalid_kernel')
        self.assertEqual(diagnostics.reason(dict(kernel_pixels=5, coverage=.2), [], [.8], .5), 'weak_response')
        evidence = dict(kernel_pixels=5, coverage=.6)
        self.assertEqual(diagnostics.reason(evidence, [.8], [.8], .5), 'ambiguous_matching')
        self.assertEqual(diagnostics.reason(evidence, [], [.8], .5), 'score_filtering')
        self.assertEqual(diagnostics.reason(evidence, [.2], [.2], .5), 'geometry')
        self.assertEqual(diagnostics.reason(evidence, [], [.2], .5), 'ambiguous_score_and_geometry')
        self.assertEqual(diagnostics.reason(evidence, [], [], .5), 'contour_postprocess')

    def test_counts_must_match_per_task(self):
        tasks = {c: {'sample_id': f'{c}::a'} for c in diagnostics.CONDITIONS}
        row = {'gt_clean': {}, 'roi_clean': {}}
        reference = {'clean': [1, 2, 1, 1], 'lr2': [0, 2, 1, 0]}
        scored = {'counts': [1, 2, 1, 1], 'matched_ann_ids': ['a_word']}
        with (patch.object(diagnostics, 'forward', return_value=(None, None, None, None, None, {}, 0)),
              patch('lowres.evaluate_shared.evaluate_image', return_value=scored)):
            with self.assertRaisesRegex(ValueError, 'job 12396 for lr2::a'):
                diagnostics.infer_pair(None, None, tasks, row, {}, reference)

    def test_paired_transition_and_6px_stratum(self):
        weak = dict(source_image_id='a', ann_id='w1', native_height=6., kernel_pixels=8, primary='weak_response',
                    sensitivity='weak_response', coverage_clean=.7)
        score = dict(weak, ann_id='w2', native_height=5., primary='score_filtering')
        records = [dict(source_image_id='a', misses=[weak, score],
                        transitions=dict(clean_tp_lr2_tp=1, clean_tp_lr2_fn=2, clean_fn_lr2_tp=1, clean_fn_lr2_fn=1))]
        summary = diagnostics.summarize(records)
        self.assertEqual(summary['transitions']['clean_fn_lr2_fn'], 1)
        self.assertEqual(summary['primary_counts'], {'weak_response': 1, 'score_filtering': 1})
        self.assertEqual(summary['sensitivity_counts'], {'weak_response': 2})
        self.assertEqual(summary['by_native_height']['4–<6'], {'score_filtering': 1})
        self.assertEqual(summary['weak_at_least_6px_clean_coverage_ge_050'], 1)
        self.assertEqual(diagnostics.representative_cases(records), [weak, score])
