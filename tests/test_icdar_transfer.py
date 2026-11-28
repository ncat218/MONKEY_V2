"""Semantic checks for the separate ICDAR evaluator and frozen experiment."""
import unittest
import numpy as np

from rev4.icdar_transfer import (CELLS, GROUPS, RECIPE, evaluate, height_bin,
                                 match_matrix, rates, vector)


class ICDARTests(unittest.TestCase):
    def test_strict_iou_and_vanilla_not_max_matching(self):
        self.assertEqual(match_matrix(np.array([[.5]])), [])
        self.assertEqual(match_matrix(np.array([[.50000001]])), [0])
        # Greedy vanilla has one match; maximum-cardinality could have two.
        self.assertEqual(match_matrix(np.array([[.8, .7], [.8, 0.]])), [0])

    def test_original_size_bins_do_not_use_upscaled_canvas(self):
        for h, expected in [(7.99,0),(8.,1),(15.99,1),(16.,2),(31.99,2),(32.,3)]:
            self.assertEqual(height_bin([0,0,10,0,10,h,0,h]), expected)

    def test_official_ignore_and_one_to_one_matching(self):
        from mmocr.evaluation.metrics import HmeanIOUMetric
        import copy
        rect = lambda x: [x,0,x+10,0,x+10,10,x,10]
        row = dict(instances=[dict(polygon=rect(0),ignore=False),dict(polygon=rect(30),ignore=True)])
        # Duplicate -> FP, ignored text -> ignored prediction, disjoint -> FP.
        pred = dict(polygons=[rect(0),rect(0),rect(30),rect(60)],scores=[.8,.8,.8,.8])
        actual = evaluate(pred,row)
        self.assertEqual(actual['counts'],[1,1,3])
        self.assertEqual(actual['bins'],[[0,0],[1,1],[0,0],[0,0]])
        metric = HmeanIOUMetric(strategy='vanilla')
        metric.pred_score_thrs = np.array([0.])
        metric.process([], [dict(pred_instances=dict(polygons=[np.array(p) for p in pred['polygons']],
                                                      scores=np.array(pred['scores'])),
                                 gt_instances=dict(polygons=[np.array(g['polygon']) for g in row['instances']],
                                                   ignored=np.array([False,True])))])
        official = metric.compute_metrics(metric.results)
        summary = rates(vector(actual))
        for key in ('precision','recall','hmean'): self.assertAlmostEqual(official[key],summary[key])

    def test_low_raw_score_already_selected_by_Q_is_not_filtered_again(self):
        row = dict(instances=[dict(polygon=[0,0,20,0,20,10,0,10],ignore=False)])
        p = dict(polygons=[[0,0,20,0,20,10,0,10]],scores=[.22])
        self.assertEqual(evaluate(p,row)['counts'],[1,1,1])

    def test_frozen_complete_design(self):
        self.assertEqual(len(CELLS),len(set(CELLS)))
        self.assertEqual(len(CELLS),12)
        for c in ['B0/reproduce','B-Clean/native/F','B1/native/F','ASF/native/F',
                  'B1/edsr_x2/F','ASF/edsr_x2/F','ASF/edsr_x2/Q',
                  'ASF/bicubic_x2/F','Nearest/native/F']: self.assertIn(c,CELLS)
        self.assertTrue(RECIPE['no_training'] and RECIPE['no_threshold_selection'])
        self.assertEqual(RECIPE['B0_reproduce']['expected'],[1800,2077,1978])

    def test_empty_tiny_is_NA_not_zero_performance(self):
        result = rates([1,1,1,0,0,1,1,0,0,0,0])
        self.assertIsNone(result['bins'][0]['recall'])
        self.assertEqual(result['bins'][1]['recall'],1.)


if __name__ == '__main__': unittest.main()
