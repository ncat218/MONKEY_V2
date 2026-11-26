"""Query-free zoom: windows from weak detector evidence and size-gated fusion.

Pure geometry/selection/fusion helpers (CPU only) so they can be unit-tested without a GPU.
All prediction polygons handled here are already in CLEAN image coordinates; window rectangles
are given in task-image pixels and converted with ``to_clean`` (= clean_size / task_size).
"""
import math

import numpy as np
from shapely.geometry import Polygon

BIN_EDGES = ((0, 4), (4, 8), (8, 16), (16, math.inf))


def height_bin(value):
    for index, (lo, hi) in enumerate(BIN_EDGES):
        if lo <= value < hi:
            return index
    return len(BIN_EDGES)-1


def ref_ratio(clean_size):
    """Clean pixels -> LR2-referenced pixels on the y axis, exactly as the GT height bins."""
    return max(1, clean_size[1]//2)/clean_size[1]


def short_side(points):
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return 0.0
    _, (w, h), _ = cv2.minAreaRect(pts)
    return float(min(w, h))


def make_preds(prediction, raw_scores, ratio, source, window=-1):
    require_equal(len(prediction['polygons']), len(raw_scores), 'Prediction/raw score lengths differ')
    preds = []
    for coords, packed, raw in zip(prediction['polygons'], prediction['scores'], raw_scores):
        points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        preds.append(dict(points=points, score=float(raw), packed=float(packed),
                          h_ref=short_side(points)*ratio, source=source, window=window))
    return preds


def shift_scale(prediction, origin, to_clean):
    """Crop-pixel polygons (already divided by any restoration factor) -> clean coordinates."""
    origin = np.asarray(origin, dtype=np.float64)
    scale = np.asarray(to_clean, dtype=np.float64)
    polygons = [((np.asarray(c, dtype=np.float64).reshape(-1, 2)+origin)*scale).reshape(-1).tolist()
                for c in prediction['polygons']]
    return dict(polygons=polygons, scores=list(prediction['scores']))


def rect_to_clean(rect, to_clean):
    x0, y0, x1, y1 = rect
    return (x0*to_clean[0], y0*to_clean[1], x1*to_clean[0], y1*to_clean[1])


def fully_inside(points, rect, image_rect, margin):
    """True when the polygon lies inside ``rect`` shrunk by ``margin`` on inner (non-image) borders."""
    x0, y0, x1, y1 = rect
    ix0, iy0, ix1, iy1 = image_rect
    lo_x = -math.inf if x0 <= ix0+1e-6 else x0+margin
    lo_y = -math.inf if y0 <= iy0+1e-6 else y0+margin
    hi_x = math.inf if x1 >= ix1-1e-6 else x1-margin
    hi_y = math.inf if y1 >= iy1-1e-6 else y1-margin
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return bool(pts[:, 0].min() >= lo_x and pts[:, 0].max() <= hi_x and
                pts[:, 1].min() >= lo_y and pts[:, 1].max() <= hi_y)


def tile_windows(width, height):
    hw, hh = width//2, height//2
    return [(0, 0, hw, hh), (hw, 0, width, hh), (0, hh, hw, height), (hw, hh, width, height)]


def candidate_windows(width, height, positions, min_px):
    ww, wh = min(width, max(min_px, width//2)), min(height, max(min_px, height//2))
    xs = sorted({int(round(v)) for v in np.linspace(0, width-ww, positions)})
    ys = sorted({int(round(v)) for v in np.linspace(0, height-wh, positions)})
    return [(x, y, x+ww, y+wh) for y in ys for x in xs]


def contains(window, box):
    return (box[0] >= window[0] and box[1] >= window[1] and box[2] <= window[2] and box[3] <= window[3])


def select_windows(boxes, width, height, k_max, positions, min_px):
    """Greedy cover of evidence boxes (task px) by at most k_max half-size windows."""
    candidates = candidate_windows(width, height, positions, min_px)
    remaining = list(range(len(boxes)))
    chosen = []
    while remaining and len(chosen) < k_max:
        best, hits = None, []
        for window in candidates:
            if window in chosen:
                continue
            inside = [i for i in remaining if contains(window, boxes[i])]
            if len(inside) > len(hits):
                best, hits = window, inside
        if not hits:
            break
        chosen.append(best)
        hit = set(hits)
        remaining = [i for i in remaining if i not in hit]
    return chosen


def pmap_evidence(probability, img_shape, scale_factor, task_to_ref, params):
    """Weak or small text-like components of the global P-map, as task-pixel boxes."""
    import cv2

    h, w = (int(v) for v in img_shape)
    p = np.asarray(probability, dtype=np.float32)[:h, :w]
    mask = (p > params['evidence_low']).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sx, sy = (float(v) for v in scale_factor)
    tau = params['tau_ref']
    boxes = []
    for label in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        if area < 4:
            continue
        h_ref = min(bw/sx, bh/sy)*task_to_ref
        if h_ref >= 2*tau:
            continue
        region = p[y:y+bh, x:x+bw][labels[y:y+bh, x:x+bw] == label]
        peak = float(region.max())
        if h_ref < tau or peak < params['evidence_strong']:
            boxes.append((x/sx, y/sy, (x+bw)/sx, (y+bh)/sy))
    return boxes


def gt_evidence(row, task_size, tau):
    """Oracle: small valid GT words (LR2-referenced height < tau), as task-pixel boxes."""
    from shapely.geometry import shape

    clean_w, clean_h = row['clean_size']
    ratio = ref_ratio(row['clean_size'])
    fx, fy = task_size[0]/clean_w, task_size[1]/clean_h
    boxes = []
    for entry in row['gt_clean']['valid']:
        x0, y0, x1, y1 = shape(entry['geometry']).bounds
        if (y1-y0)*ratio < tau:
            boxes.append((x0*fx, y0*fy, x1*fx, y1*fy))
    return boxes


def _polygon(points):
    polygon = Polygon(np.asarray(points, dtype=np.float64).reshape(-1, 2))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _bounds(preds):
    if not preds:
        return np.zeros((0, 4))
    return np.asarray([[p['points'][:, 0].min(), p['points'][:, 1].min(),
                        p['points'][:, 0].max(), p['points'][:, 1].max()] for p in preds])


def iou(a, b):
    pa, pb = _polygon(a['points']), _polygon(b['points'])
    if pa.is_empty or pb.is_empty:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area+pb.area-inter
    return float(inter/union) if union > 0 else 0.0


def _overlaps(box_a, box_b):
    return not (box_a[0] > box_b[2] or box_b[0] > box_a[2] or box_a[1] > box_b[3] or box_b[1] > box_a[3])


def add_without_overlap(kept, candidates, threshold):
    """Keep every box in ``kept``; add candidates in score order when IoU < threshold with all kept boxes.

    Zoom boxes from overlapping windows are de-duplicated against each other the same way. A kept
    (global) box is never removed by a candidate.
    """
    result = list(kept)
    bounds = [b for b in _bounds(result)]
    for cand in sorted(candidates, key=lambda p: -p['score']):
        cb = _bounds([cand])[0]
        if any(_overlaps(cb, b) and iou(cand, p) >= threshold for p, b in zip(result, bounds)):
            continue
        result.append(cand)
        bounds.append(cb)
    return result


def fuse(global_preds, window_preds, windows, image_rect, mode, cutoff, params):
    """Combine one global pass with zoom passes.

    replace: inside a window the zoom pass owns small words: small global boxes fully inside a window
             are dropped and small zoom boxes are added unless they overlap a remaining box;
    union:   keep every global box and add small zoom boxes that do not overlap any kept box (ZB1-style);
    verify:  keep global boxes, but a small global box fully inside a window survives only if the zoom
             pass also finds it (IoU >= verify_iou); no zoom box is added.
    """
    tau, margin = params['tau_ref'], params['margin_clean']
    base = [p for p in global_preds if p['score'] >= cutoff]
    zoom = [[z for z in preds if z['score'] >= cutoff and fully_inside(z['points'], windows[w], image_rect, margin)]
            for w, preds in enumerate(window_preds)]

    def covered(p):
        return any(fully_inside(p['points'], rect, image_rect, margin) for rect in windows)

    if mode == 'verify':
        flat = [z for group in zoom for z in group]
        boxes = _bounds(flat)
        kept = []
        for p in base:
            if p['h_ref'] < tau and covered(p):
                px0, py0 = p['points'][:, 0].min(), p['points'][:, 1].min()
                px1, py1 = p['points'][:, 0].max(), p['points'][:, 1].max()
                near = [z for z, b in zip(flat, boxes) if not (b[0] > px1 or px0 > b[2] or b[1] > py1 or py0 > b[3])]
                if not any(iou(p, z) >= params['verify_iou'] for z in near):
                    continue
            kept.append(p)
        return kept
    small = [z for group in zoom for z in group if z['h_ref'] < tau]
    if mode == 'replace':
        base = [p for p in base if not (p['h_ref'] < tau and covered(p))]
    else:
        require_equal(mode, 'union', 'Unknown fusion mode')
    return add_without_overlap(base, small, params['nms_iou'])


def score_detail(matcher, cutoff, heights):
    """CandidateMatcher.score plus FP counts by predicted LR2-referenced height bin.

    Returns [tp, gt, pred, gt0, tp0, gt1, tp1, gt2, tp2, gt3, tp3, fp0, fp1, fp2, fp3].
    """
    from lowres.evaluate_shared import maximum_matching

    kept = [p for p in matcher.originals if matcher.raw_scores[p] >= cutoff]
    to_local = {p: i for i, p in enumerate(kept)}
    adjacency = [[to_local[p] for p in edge if p in to_local] for edge in matcher.edges]
    pairs = maximum_matching(adjacency, len(kept))
    gt_hit = {g for g, _ in pairs}
    pred_hit = {kept[local] for _, local in pairs}
    invalid = [p for p in matcher.invalid if matcher.raw_scores[p] >= cutoff]
    values = [len(pairs), len(matcher.ann_ids), len(kept)+len(invalid)]
    for lo, hi in BIN_EDGES:
        group = {i for i, h in enumerate(matcher.heights) if lo <= h < hi}
        values += [len(group), len(group & gt_hit)]
    fp_bins = [0]*len(BIN_EDGES)
    for p in [q for q in kept if q not in pred_hit]+invalid:
        fp_bins[height_bin(heights[p])] += 1
    return values+fp_bins


def require_equal(left, right, message):
    if left != right:
        raise RuntimeError(f'{message}: {left!r} != {right!r}')
