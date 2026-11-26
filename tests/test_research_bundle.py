"""CPU checks for the research bundle: plan, zoom geometry/fusion, detailed scoring, report helpers, PRC2 selection."""
import importlib.util
import json
import math
from unittest import TestCase, skipUnless

import numpy as np
from shapely.geometry import box, mapping

from rev4 import zoom_core as Z
from rev4.bundle_plan import (GRID, PLAIN_VARIANTS, PRC2, SETTINGS, STRESS_VARIANTS, TRAIN_ARMS, ZOOM, ZOOM_MODELS,
                              ZOOM_RULES, ZOOM_VARIANTS, eval_settings, variants_for)

POLICY = dict(match_iou=.5, min_prediction_roi_fraction=.5, ignore_prediction_overlap=.5, word_coverage=.5)


def quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def pred(x0, y0, x1, y1, score=.9, source='global', ratio=.5):
    points = quad(x0, y0, x1, y1)
    return dict(points=points, score=score, packed=score, h_ref=Z.short_side(points)*ratio, source=source, window=-1)


class PlanTests(TestCase):
    def test_plan_is_consistent(self):
        self.assertEqual(TRAIN_ARMS, ('B1_s43', 'BClean_s43', 'PRC2'))
        self.assertIn('0.40/0.60', eval_settings())
        self.assertEqual(len(GRID), 15)
        self.assertEqual(SETTINGS, {'primary': (.40, .60), 'common': (.30, .40)})
        self.assertEqual(set(ZOOM_MODELS), set(ZOOM_VARIANTS))
        for variants in ZOOM_VARIANTS.values():
            self.assertEqual(variants[0], 'global')
            self.assertEqual(len(set(variants)), len(variants))
            for variant in variants:
                self.assertTrue(variant in ZOOM_RULES or variant in PLAIN_VARIANTS, variant)
        self.assertTrue(set(STRESS_VARIANTS) <= set(ZOOM_VARIANTS['B1']))
        self.assertNotIn('guided_edsr', ZOOM_VARIANTS['B-Clean'])
        for source, mode, restoration in ZOOM_RULES.values():
            self.assertIn(source, ('tile', 'guided', 'oracle'))
            self.assertIn(mode, ('replace', 'union', 'verify'))
            self.assertIn(restoration, ('none', 'edsr'))

    def test_variants_by_condition(self):
        self.assertEqual(variants_for('B1', 'lr2'), ZOOM_VARIANTS['B1'])
        self.assertEqual(variants_for('B-Clean', 'clean'), ZOOM_VARIANTS['B-Clean'])
        self.assertEqual(variants_for('B1', 'jpeg10'), STRESS_VARIANTS)
        with self.assertRaises(RuntimeError):
            variants_for('B-Clean', 'lr4')

    def test_identity_round_trips_through_json(self):
        # Launch / record identities are compared after a JSON round trip: no tuples allowed inside.
        for value in (ZOOM, PRC2, {k: list(v) for k, v in ZOOM_RULES.items()}):
            self.assertEqual(json.loads(json.dumps(value)), value)

    def test_zoom_study_scores_both_fixed_points(self):
        from rev4 import zoom_study as S

        self.assertEqual(S.PAIRS, ((.40, .60), (.30, .40)))
        self.assertEqual(S.MASKS, (.30, .40))
        self.assertEqual((S.PRIMARY, S.COMMON), ('0.40/0.60', '0.30/0.40'))


class WindowTests(TestCase):
    def test_tiles_partition_the_image(self):
        tiles = Z.tile_windows(101, 77)
        area = sum((x1-x0)*(y1-y0) for x0, y0, x1, y1 in tiles)
        self.assertEqual(area, 101*77)

    def test_greedy_selection_covers_evidence_with_half_windows(self):
        boxes = [(5, 5, 10, 8), (12, 6, 18, 9), (90, 60, 95, 64)]
        windows = Z.select_windows(boxes, 100, 80, 4, 5, 16)
        self.assertEqual(len(windows), 2)
        for window in windows:
            self.assertEqual((window[2]-window[0], window[3]-window[1]), (50, 40))
        self.assertTrue(all(any(Z.contains(w, b) for w in windows) for b in boxes))

    def test_window_cap_is_respected(self):
        boxes = [(x, y, x+2, y+2) for x in (1, 30, 60, 90) for y in (1, 40, 75)]
        windows = Z.select_windows(boxes, 100, 80, 2, 5, 16)
        self.assertEqual(len(windows), 2)

    def test_no_evidence_no_window(self):
        self.assertEqual(Z.select_windows([], 100, 80, 4, 5, 16), [])

    def test_pmap_evidence_keeps_weak_or_small_components(self):
        p = np.zeros((100, 200), dtype=np.float32)
        p[10:13, 10:30] = .35          # small, weak: 3 px tall
        p[40:80, 20:180] = .95         # large and confident
        p[85:95, 150:190] = .45        # medium height, weak peak
        params = dict(ZOOM)
        boxes = Z.pmap_evidence(p, (100, 200), (1., 1.), 1., params)
        tops = sorted(round(b[1]) for b in boxes)
        self.assertEqual(tops, [10, 85])

    def test_pmap_evidence_maps_network_pixels_to_task_pixels(self):
        p = np.zeros((100, 200), dtype=np.float32)
        p[20:24, 40:60] = .5           # 4 network px tall at scale 2 -> 2 task px
        boxes = Z.pmap_evidence(p, (100, 200), (2., 2.), 1., dict(ZOOM))
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], (20, 10, 30, 12))


class GeometryTests(TestCase):
    def test_shift_scale_maps_crop_pixels_to_clean(self):
        prediction = dict(polygons=[[0, 0, 4, 0, 4, 2, 0, 2]], scores=[.8])
        out = Z.shift_scale(prediction, (10, 20), (2., 2.))
        np.testing.assert_allclose(out['polygons'][0], [20, 40, 28, 40, 28, 44, 20, 44])

    def test_inside_ignores_margin_on_image_border(self):
        image = (0, 0, 100, 80)
        window = (0, 0, 50, 40)
        self.assertTrue(Z.fully_inside(quad(0, 0, 10, 5), window, image, 1.))
        self.assertFalse(Z.fully_inside(quad(45, 5, 49.5, 9), window, image, 1.))
        self.assertTrue(Z.fully_inside(quad(45, 5, 48.5, 9), window, image, 1.))

    def test_reference_ratio_matches_native_lr2_height(self):
        self.assertEqual(Z.ref_ratio((1024, 768)), .5)
        self.assertAlmostEqual(Z.ref_ratio((1001, 767)), 383/767)


class FusionTests(TestCase):
    params = dict(ZOOM, margin_clean=1.)
    image = (0., 0., 200., 160.)
    window = [(0., 0., 100., 80.)]

    def test_replace_moves_small_words_to_the_zoom_pass(self):
        global_preds = [pred(10, 10, 30, 14), pred(120, 100, 190, 140)]
        zoom = [[pred(11, 10, 30, 14, .7, 'zoom'), pred(40, 30, 60, 34, .8, 'zoom')]]
        fused = Z.fuse(global_preds, zoom, self.window, self.image, 'replace', .6, self.params)
        sources = sorted((p['source'], round(p['points'][0, 0])) for p in fused)
        self.assertEqual(sources, [('global', 120), ('zoom', 11), ('zoom', 40)])

    def test_union_keeps_global_and_suppresses_duplicate_zoom(self):
        global_preds = [pred(10, 10, 30, 14)]
        zoom = [[pred(10, 10, 30, 14, .7, 'zoom'), pred(40, 30, 60, 34, .8, 'zoom')]]
        fused = Z.fuse(global_preds, zoom, self.window, self.image, 'union', .6, self.params)
        self.assertEqual(sorted(p['source'] for p in fused), ['global', 'zoom'])

    def test_verify_drops_unconfirmed_small_global_boxes_only(self):
        global_preds = [pred(10, 10, 30, 14), pred(50, 50, 70, 54), pred(120, 100, 190, 140)]
        zoom = [[pred(10, 10, 30, 14, .7, 'zoom')]]
        fused = Z.fuse(global_preds, zoom, self.window, self.image, 'verify', .6, self.params)
        self.assertEqual(sorted(round(p['points'][0, 0]) for p in fused), [10, 120])

    def test_zoom_boxes_on_inner_window_border_are_dropped(self):
        zoom = [[pred(90, 10, 99.5, 14, .9, 'zoom')]]
        fused = Z.fuse([], zoom, self.window, self.image, 'union', .6, self.params)
        self.assertEqual(fused, [])

    def test_large_zoom_boxes_are_not_added(self):
        zoom = [[pred(10, 10, 60, 50, .9, 'zoom')]]
        fused = Z.fuse([], zoom, self.window, self.image, 'replace', .6, self.params)
        self.assertEqual(fused, [])

    def test_scores_below_cutoff_are_removed(self):
        fused = Z.fuse([pred(10, 10, 30, 14, .5)], [[]], self.window, self.image, 'replace', .6, self.params)
        self.assertEqual(fused, [])

    def test_kept_boxes_are_never_removed(self):
        kept = [pred(0, 0, 10, 4), pred(0, 0, 10, 4.2)]
        result = Z.add_without_overlap(kept, [pred(0, 0, 10, 4.1, .99, 'zoom')], .5)
        self.assertEqual([p['source'] for p in result], ['global', 'global'])

    def test_duplicate_zoom_boxes_keep_the_higher_score(self):
        candidates = [pred(40, 30, 60, 34, .7, 'zoom'), pred(40.5, 30, 60, 34, .9, 'zoom'), pred(70, 30, 90, 34, .8, 'zoom')]
        result = Z.add_without_overlap([], candidates, .5)
        self.assertEqual(sorted(p['score'] for p in result), [.8, .9])


class ScoringTests(TestCase):
    def test_detail_matches_candidate_matcher(self):
        from rev4.decision_core import CandidateMatcher

        row = dict(clean_size=[200, 160], roi_clean=mapping(box(0, 0, 200, 160)),
                   gt_clean=dict(valid=[dict(ann_id='a', geometry=mapping(box(10, 10, 30, 14))),
                                        dict(ann_id='b', geometry=mapping(box(50, 50, 90, 80)))], ignored=[]))
        preds = [pred(10, 10, 30, 14), pred(120, 100, 130, 103), pred(50, 50, 90, 80, .55)]
        prediction = dict(polygons=[p['points'].reshape(-1).tolist() for p in preds], scores=[p['packed'] for p in preds])
        matcher = CandidateMatcher(prediction, [p['score'] for p in preds], row, POLICY)
        detail = Z.score_detail(matcher, .6, [p['h_ref'] for p in preds])
        fast = matcher.score(.6)
        self.assertEqual(detail[:3], fast['counts'])
        self.assertEqual(detail[3:11], [x for pair in fast['bins'] for x in pair])
        self.assertEqual(detail[:3], [1, 2, 2])
        self.assertEqual(sum(detail[11:]), 1)
        self.assertEqual(detail[11], 1)


class ReportTests(TestCase):
    def test_fusion_counts_layout(self):
        from rev4.bundle_report import detail_from_counts

        pairs = ((.3, .4), (.4, .3), (.4, .4), (.4, .5), (.4, .6), (.4, .7))
        task = dict(counts=[[0, 0, 0]]*4+[[5, 9, 7], [0, 0, 0]],
                    bins=[[[0, 0]]*4]*4+[[[1, 0], [2, 1], [3, 2], [3, 2]], [[0, 0]]*4])
        self.assertEqual(detail_from_counts(task, '0.40/0.60', pairs), [5, 9, 7, 1, 0, 2, 1, 3, 2, 3, 2])
        bundle = dict(counts={'0.40/0.60': [5, 9, 7]}, bins={'0.40/0.60': [1, 0, 2, 1, 3, 2, 3, 2]})
        self.assertEqual(detail_from_counts(bundle, '0.40/0.60'), [5, 9, 7, 1, 0, 2, 1, 3, 2, 3, 2])

    def test_compare_is_paired_and_reports_small_text(self):
        from rev4.bundle_report import compare

        old = np.array([[5, 10, 8, 4, 1, 2, 1, 2, 1, 2, 2]]*20, dtype=np.int64)
        same = compare(old, old.copy())
        self.assertEqual((same['hmean']['delta'], same['tiny']['delta'], same['fp']['delta']), (0., 0., 0))
        self.assertEqual(same['tiny']['ci'], [0., 0.])
        new = old.copy()
        new[:, 0] += 1
        new[:, 4] += 1
        new[:, 2] += 1
        better = compare(old, new)
        self.assertAlmostEqual(better['tiny']['delta'], 25.)
        self.assertAlmostEqual(better['lt8']['delta'], 100*(1/6))
        self.assertEqual(better['fp']['delta'], 0)
        self.assertGreater(better['hmean']['ci'][0], 0)

    def test_matched_fp_lookup_does_not_extrapolate(self):
        from rev4.bundle_report import at_fp

        front = np.array([[100., .30, .50, .70], [200., .40, .60, .75]])
        self.assertAlmostEqual(at_fp(front, 150, 1), .35)
        self.assertTrue(math.isnan(at_fp(front, 50, 1)))
        self.assertTrue(math.isnan(at_fp(front, 250, 1)))
        self.assertTrue(math.isnan(at_fp(None, 150, 1)))


@skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('mmocr'), 'Requires torch and mmocr')
class PRC2Tests(TestCase):
    def test_kernel_selection_uses_teacher_coverage(self):
        import torch
        from rev4.prc2_plugins import kernel_selection

        labels = torch.zeros((6, 8), dtype=torch.long)
        labels[1:3, 1:4] = 1   # 6 px kernel, teacher covers it
        labels[4:6, 5:8] = 2   # 6 px kernel, teacher misses it
        labels[0, 7] = 3       # 1 px kernel, too small
        probability = torch.zeros((6, 8))
        probability[1:3, 1:4] = .9
        probability[0, 7] = .9
        valid = torch.ones((6, 8), dtype=torch.bool)
        mask, selected, eligible = kernel_selection(probability, labels, valid, .3, .3, 4)
        self.assertEqual((selected, eligible), (1, 2))
        self.assertEqual(int(mask.sum()), 6)
        self.assertTrue(bool(mask[1:3, 1:4].all()))
        valid[1:3, 1:4] = False
        mask, selected, eligible = kernel_selection(probability, labels, valid, .3, .3, 4)
        self.assertEqual((selected, eligible, int(mask.sum())), (0, 1, 0))
