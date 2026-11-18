"""Read-only instrumentation and indexed evaluation around frozen helpers."""
import copy

import numpy as np
from shapely.geometry import shape

from lowres.evaluate_shared import evaluate_image, maximum_matching
from rev4 import require
from rev4.paired_eda_core import filtered_geometry, overlap_index, overlaps


def select_candidates(prediction, raw_scores, cutoff):
    # Native DBPostprocessor compares the Python score BEFORE casting to FP32.
    require(len(raw_scores) == len(prediction['scores']), 'Raw/packed score count mismatch')
    indices = [i for i, score in enumerate(raw_scores) if score >= cutoff]
    return dict(polygons=[prediction['polygons'][i] for i in indices],
                scores=[prediction['scores'][i] for i in indices])


def capture_candidates(native, probability, sample, restore, geometry):
    """Run original installed postprocessor; observe its scoring/unclip helpers."""
    import cv2
    from lowres.inference import unpack

    name, mask, ratio, width, epsilon = geometry
    processor = copy.copy(native)
    processor.mask_thr, processor.unclip_ratio = mask, ratio
    processor.min_text_width, processor.epsilon_ratio = width, epsilon
    processor.min_text_score = 0.
    require(processor.text_repr_type == 'quad', 'Only the frozen quad path is supported')
    score_method, unclip_method = processor._get_bbox_score, processor._unclip
    events = []

    def observe_score(score_map, polygon):
        score = float(score_method(score_map, polygon))
        require(np.isfinite(score) and 0 <= score <= 1, 'Invalid raw native score')
        events.append(dict(raw_score=score, vertices_before_unclip=len(polygon), status='scored'))
        return score

    def observe_unclip(polygon):
        result = unclip_method(polygon)
        event = events[-1]
        require(event['status'] == 'scored', 'Unexpected native unclip ordering')
        if len(result) == 0:
            event['status'] = 'unclip_empty'
        else:
            rectangle = cv2.minAreaRect(result.reshape(-1, 2))
            event['short_edge_network_px'] = float(min(rectangle[1]))
            event['status'] = 'accepted' if min(rectangle[1]) >= width else 'width_rejected'
        return result

    processor._get_bbox_score, processor._unclip = observe_score, observe_unclip
    prediction = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
    accepted = [event['raw_score'] for event in events if event['status'] == 'accepted']
    require(len(accepted) == len(prediction['polygons']), 'Observed native output count changed')
    contours, _ = cv2.findContours(np.uint8(probability[0].detach().cpu().numpy() > mask)*255,
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    examined = min(len(contours), native.max_candidates+1)  # Mirror native i > max_candidates.
    require(examined >= len(events), 'Native contour instrumentation mismatch')
    return prediction, accepted, dict(geometry=name, contours=len(contours), examined=examined,
            candidate_limit_skipped=len(contours)-examined, approximation_rejected=examined-len(events),
            events=events)


class CandidateMatcher:
    """Exact TP/GT/pred matching, omitting expensive diagnostic coverage unions."""
    def __init__(self, prediction, raw_scores, row, policy):
        self.prediction, self.raw_scores = prediction, raw_scores
        self.row, self.policy = row, policy
        self.geometries, self.invalid = filtered_geometry(prediction, row, policy)
        self.originals = list(self.geometries)
        self.ann_ids = [str(entry['ann_id']) for entry in row['gt_clean']['valid']]
        self.heights = np.array([(shape(entry['geometry']).bounds[3]-shape(entry['geometry']).bounds[1])*
                                (max(1, row['clean_size'][1]//2)/row['clean_size'][1])
                                for entry in row['gt_clean']['valid']])
        index = overlap_index(self.geometries, prediction['scores'])
        self.edges = [[p for iou, _, p in overlaps(index, shape(entry['geometry']))
                       if iou >= policy['match_iou']] for entry in row['gt_clean']['valid']]

    def score(self, cutoff):
        kept = [p for p in self.originals if self.raw_scores[p] >= cutoff]
        to_local = {p: i for i, p in enumerate(kept)}
        adjacency = [[to_local[p] for p in edge if p in to_local] for edge in self.edges]
        pairs = maximum_matching(adjacency, len(kept))
        gt_indices = {g for g, _ in pairs}
        invalid_count = sum(self.raw_scores[p] >= cutoff for p in self.invalid)
        bins = []
        for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf'))):
            group = {i for i, h in enumerate(self.heights) if lo <= h < hi}
            bins.append([len(group), len(group & gt_indices)])
        return dict(counts=[len(pairs), len(self.ann_ids), len(kept)+invalid_count], bins=bins,
                    matched_ann_ids=[self.ann_ids[i] for i in sorted(gt_indices)])

    def assert_official(self, cutoff):
        prediction = select_candidates(self.prediction, self.raw_scores, cutoff)
        official = evaluate_image(self.row['gt_clean'], self.row['roi_clean'], prediction, self.policy)
        fast = self.score(cutoff)
        require(fast['counts'] == official['counts'][:3] and
                fast['matched_ann_ids'] == list(map(str, official['matched_ann_ids'])),
                'Indexed TP/matching differs from unchanged evaluator')
        return fast


def detection_rates(counts):
    tp, gt, pred = map(int, counts)
    return dict(tp=tp, fp=pred-tp, fn=gt-tp, precision=tp/pred if pred else 0.,
                recall=tp/gt if gt else 0., hmean=2*tp/(gt+pred) if gt+pred else 0.)


def choose_setting(records):
    from fractions import Fraction

    def rank(record):
        value = sum(Fraction(2*record[c]['tp'], 2*record[c]['tp']+record[c]['fp']+record[c]['fn'] or 1)
                    for c in ('clean', 'lr2'))
        params = record['setting']
        native = params['geometry'] == 'base' and params['min_text_score'] == .3
        return value, native
    return max(records, key=rank)


def summarize_counts(matrix):
    """Aggregate [source, condition, tp/gt/pred, tiny_gt/tiny_tp] counts."""
    import numpy as np

    total = np.asarray(matrix, dtype=np.int64).sum(axis=0)
    result = {}
    for index, condition in enumerate(('clean', 'lr2')):
        tp, gt, pred, tiny_gt, tiny_tp = map(int, total[index])
        result[condition] = dict(tp=tp, gt=gt, pred=pred, fp=pred-tp, fn=gt-tp,
            precision=tp/pred if pred else 0., recall=tp/gt if gt else 0.,
            hmean=2*tp/(gt+pred) if gt+pred else 0., tiny_gt=tiny_gt,
            tiny_tp=tiny_tp, tiny_recall=tiny_tp/tiny_gt if tiny_gt else 0.)
    return result


def paired_hmean_ci(base, candidate, condition_index, indices, repetitions=2000):
    """Source-cluster bootstrap for a candidate minus base Hmean difference."""
    import numpy as np

    base = np.asarray(base, dtype=np.int64)[:, condition_index, :3]
    candidate = np.asarray(candidate, dtype=np.int64)[:, condition_index, :3]
    sampled_base = base[indices].sum(axis=1)
    sampled_candidate = candidate[indices].sum(axis=1)
    base_h = 2*sampled_base[:, 0]/np.maximum(sampled_base[:, 1]+sampled_base[:, 2], 1)
    candidate_h = 2*sampled_candidate[:, 0]/np.maximum(sampled_candidate[:, 1]+sampled_candidate[:, 2], 1)
    delta = 100*(candidate_h-base_h)
    observed_base = base.sum(axis=0)
    observed_candidate = candidate.sum(axis=0)
    point = 100*(2*observed_candidate[0]/max(observed_candidate[1]+observed_candidate[2], 1) -
                 2*observed_base[0]/max(observed_base[1]+observed_base[2], 1))
    return dict(repetitions=repetitions, delta_pp=float(point),
                ci95_pp=np.quantile(delta, [.025, .975]).tolist(), bootstrap_mean_pp=float(delta.mean()))
