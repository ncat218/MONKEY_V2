"""Learned box-quality scoring from ONE B1 forward: P0 inputs, smoke/extract (GPU), fit + calibrate (CPU).
Spec (locked before any GPU run): docs/research/janek/boxquality_spec_20261124_janek.md. NOT an improvement claim.

Commands
  rows      build TRAIN label rows (TextOCR train JSON + vendored gt_for_source, ROI = full canvas)         CPU
  prepare   verify TRAIN Clean bytes, materialize native LR2 (Tuan's native_lr2, Pillow 11.3.0)              CPU
  extract   B1 forward + instrumented post-processing: candidates (raw >= .20) + features + timings         GPU
            --consistency : also run Tuan's plain capture_candidates and require identical polygons/raw
  devcheck  same extraction on a few development LR2 images; compare with the T4 dump (pipeline check only) GPU
  fit       labels (official matching), weighted L2 logistic (fixed), cutoff on TRAIN-calibration for RAW,
            SCORE-ONLY and FULL; paired bootstrap FULL - RAW; review sheet                                   CPU
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path, PureWindowsPath

from mixab import read_json, require, save_json 

TEXTOCR_TRAIN_SHA256 = '5bac8303c5999a3ddf630efde8ac9743b4461be6022b193a1075df90fbd2ce0c'   # MONKEY_V2/rev4/train_recovery.py:20
TEXTOCR_URL = 'https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json'
TEXTOCR_BYTES = 279887401
TRAIN_IDS_SHA256 = '8ac5cbd98db0ffcb67536c6629ed4839ecf8c8e82b927796d0f9208102805f60'
SEED = '20261124:boxquality-v1:'
N_SMOKE, N_SELECT, N_CAL = 10, 2000, 400
SELECTION_RULE = (f'Eligible = TRAIN sources (train_ids.csv) with >= 1 valid GT under the full-canvas rule. Order by '
                  f'sha256("{SEED}" + id). First {N_SMOKE} = smoke; next {N_SELECT} = selected. Among the selected, the '
                  f'{N_CAL} with the smallest sha256("{SEED}cal:" + id) = calibration, the rest = fit. Clean and LR2 of a '
                  f'source are always in the same fold.')
GEOM = ('mask_040', .4, 1.5, 5, .01)          # B1 .40/.60 geometry (mask .40, unclip 1.5, min width 5, eps .01)
CUT_F = .6                                    # B1 .40/.60 score cutoff (reported; RAW control re-selects its own cutoff)
FLOOR = .20                                   # candidate pool, same as the DualAgree/Flip-TTA dumps
FEATURES = ('raw', 'std', 'q10', 'q50', 'q90', 'pmax', 'ring_mean', 'quad_mean', 'fill_quad', 'fill_hull',
            'short_norm', 'log_aspect', 'n_vertices')
L2 = 1.0                                      # fixed ridge on standardized weights (intercept not penalized); no sweep
N_CUTS = 101                                  # cutoff grid = quantiles 0..100 % of the scorer on calibration candidates
TINY_PP = -3.                                 # tiny (LR2) constraint vs F .40/.60 on the same calibration sources
GATE_PP = .30                                 # FULL - RAW on calibration, mean Hmean(Clean, LR2), and CI lower > 0
BOOT_N, BOOT_SEED = 2000, 431


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _rank(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def select_train_sources(eligible):
    """eligible: dict rows with source_image_id/clean_relpath/clean_sha256 -> list with role smoke/fit/calibration."""
    order = sorted(eligible, key=lambda r: _rank(SEED + r['source_image_id']))
    require(len(order) >= N_SMOKE + N_SELECT, 'Not enough eligible TRAIN sources')
    smoke, chosen = order[:N_SMOKE], order[N_SMOKE:N_SMOKE + N_SELECT]
    cal = set(r['source_image_id'] for r in sorted(chosen, key=lambda r: _rank(SEED + 'cal:' + r['source_image_id']))[:N_CAL])
    out = [dict(source_image_id=r['source_image_id'], clean_relpath=r['clean_relpath'], clean_sha256=r['clean_sha256'], role='smoke')
           for r in smoke]
    out += [dict(source_image_id=r['source_image_id'], clean_relpath=r['clean_relpath'], clean_sha256=r['clean_sha256'],
                 role='calibration' if r['source_image_id'] in cal else 'fit') for r in chosen]
    return out


# ---------------------------------------------------------------- image fetch (official zip, per member via HTTP Range)

TEXTVQA_ZIP_URL = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
TEXTVQA_ZIP_BYTES = 7072297970


class RangeFile:
    """Minimal seekable reader over HTTP Range (used only to read the zip central directory and single members)."""

    def __init__(self, url, size, opener=None):
        import urllib.request
        self.url, self.size, self.pos, self.fetched = url, size, 0, 0
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=120))

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        import urllib.request
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}', 'User-Agent': 'cs431-boxquality'})
        for attempt in range(4):
            try:
                with self._open(req) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2*(attempt + 1))
        require(len(data) == end - self.pos + 1, 'Short HTTP range read')
        self.pos += len(data); self.fetched += len(data)
        return data


def fetch_members(items, dest, url=TEXTVQA_ZIP_URL, size=TEXTVQA_ZIP_BYTES):
    """items: [(relpath, sha256)]. Download each zip member by one Range request (local header + data), inflate,
    check CRC-32 and SHA-256, write atomically. Already-correct files are skipped. Returns dict(ok, fetched_mib)."""
    import struct
    import zipfile
    import zlib
    dest = Path(dest)
    todo = [(r, h) for r, h in items if not ((dest/r).is_file() and sha256_file(dest/r) == h)]
    if not todo:
        return dict(ok=len(items), fetched_mib=0.)
    rf = RangeFile(url, size)
    zf = zipfile.ZipFile(rf)
    info = {i.filename: i for i in zf.infolist()}
    absent = [r for r, _ in todo if r not in info]
    require(not absent, f'{len(absent)} members absent from the zip, e.g. {absent[:3]}')
    for r, h in todo:
        zi = info[r]
        rf.seek(zi.header_offset)
        head = rf.read(30)
        require(head[:4] == b'PK', f'Bad local header for {r}')
        n_name, n_extra = struct.unpack('<HH', head[26:30])
        rf.seek(zi.header_offset + 30 + n_name + n_extra)
        comp = rf.read(zi.compress_size)
        if zi.compress_type == zipfile.ZIP_STORED:
            data = comp
        elif zi.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(comp)
        else:
            raise ValueError(f'Unsupported compression {zi.compress_type} for {r}')
        require(zlib.crc32(data) & 0xffffffff == zi.CRC, f'CRC mismatch for {r}')
        require(hashlib.sha256(data).hexdigest() == h, f'SHA-256 mismatch for {r}')
        out = dest/r
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + '.part')
        tmp.write_bytes(data)
        tmp.replace(out)
    return dict(ok=len(items), fetched_mib=rf.fetched/2**20)


# ---------------------------------------------------------------- label rows (CPU)

def train_row(source, textocr, width, height, sm):
    """Row in the frozen manifest schema needed by CandidateMatcher. ROI = full canvas (native Clean/LR2 have no
    perspective warp; historical degradation metadata is HPC-only). gt via the vendored gt_for_source."""
    from shapely.geometry import box, mapping
    roi = box(0, 0, width, height)
    gt, removed = sm.gt_for_source(source, textocr, roi, sm.DEFAULT_SHARED_CONFIG, Counter())
    return dict(source_image_id=source, clean_size=[int(width), int(height)], roi_clean=mapping(roi), gt_clean=gt,
                removed_gt_ids=removed, roi_rule='full_canvas')


def rows_cmd(a):
    from mixab.vendor import shared_manifest_janek as sm 
    require(sha256_file(a.textocr) == TEXTOCR_TRAIN_SHA256, 'TextOCR train JSON SHA-256 differs from Tuan')
    require(sha256_file(a.train_ids) == TRAIN_IDS_SHA256, 'train_ids.csv SHA-256 differs')
    textocr, _ = sm.normalize_textocr(json.loads(Path(a.textocr).read_text(encoding='utf-8')))
    train = list(csv.DictReader(open(a.train_ids, encoding='utf-8')))
    eligible = []
    for t in train:
        im = textocr['imgs'].get(t['source_image_id'])
        if im is None:
            continue
        r = train_row(t['source_image_id'], textocr, im['width'], im['height'], sm)
        if r['gt_clean']['valid']:
            eligible.append(t)
    sel = select_train_sources(eligible)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out/'train_rows.jsonl.gz', 'wt', encoding='utf-8') as f:
        for x in sel:
            im = textocr['imgs'][x['source_image_id']]
            f.write(json.dumps(dict(train_row(x['source_image_id'], textocr, im['width'], im['height'], sm),
                                    clean_relpath=x['clean_relpath'], clean_sha256=x['clean_sha256'], role=x['role'])) + '\n')
    save_json(out/'rows_receipt.json', dict(train_sources=len(train), eligible=len(eligible), roles=dict(Counter(x['role'] for x in sel)),
                                            rule=SELECTION_RULE, textocr_sha256=TEXTOCR_TRAIN_SHA256))
    print(json.dumps(dict(eligible=len(eligible), roles=dict(Counter(x['role'] for x in sel)))))


def load_rows(path):
    return {r['source_image_id']: r for r in (json.loads(line) for line in gzip.open(path, 'rt', encoding='utf-8'))}


def tasks_for(rows, roles, clean_root, prepared_root):
    """[(task dict)] for Clean + native LR2 of each source with a role in `roles` (sorted, deterministic)."""
    out = []
    for s in sorted(rows):
        r = rows[s]
        if r['role'] not in roles:
            continue
        W, H = r['clean_size']
        out.append(dict(source=s, condition='clean', path=str(Path(clean_root)/r['clean_relpath']), size=[W, H],
                        sha256=r['clean_sha256'], role=r['role']))
        out.append(dict(source=s, condition='lr2', path=str(Path(prepared_root)/f'lr2/{s}.png'),
                        size=[max(1, W//2), max(1, H//2)], sha256=None, role=r['role']))
    return out


# ---------------------------------------------------------------- features (pure, CPU-testable)

def region_stats(score_map, poly_pts):
    """Statistics of P inside the SAME mask Tuan's _get_bbox_score uses (bbox-clipped fillPoly of the pre-unclip polygon)."""
    import cv2
    import numpy as np
    h, w = score_map.shape[:2]
    pts = np.asarray(poly_pts, dtype=np.float64).reshape(-1, 2).copy()
    xmin = int(np.clip(np.floor(pts[:, 0].min()), 0, w - 1)); xmax = int(np.clip(np.ceil(pts[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(pts[:, 1].min()), 0, h - 1)); ymax = int(np.clip(np.ceil(pts[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    pts[:, 0] -= xmin; pts[:, 1] -= ymin
    cv2.fillPoly(mask, pts.reshape(1, -1, 2).astype(np.int32), 1)
    vals = score_map[ymin:ymax + 1, xmin:xmax + 1][mask.astype(bool)]
    if vals.size == 0:
        return dict(mean=float(cv2.mean(score_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]), std=0., q10=0., q50=0., q90=0., pmax=0., n=0)
    q10, q50, q90 = np.percentile(vals, (10, 50, 90))
    return dict(mean=float(cv2.mean(score_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]), std=float(vals.std()),
                q10=float(q10), q50=float(q50), q90=float(q90), pmax=float(vals.max()), n=int(vals.size))


def quad_features(score_map, poly_pts, quad, tensor_short):
    """Ring/quad means and shape ratios. quad: 4x2 (network grid), the minAreaRect box of the unclipped polygon.
    Masks are built on the local bounding box of quad u polygon (clipped to the map), so cost scales with box size."""
    import cv2
    import numpy as np
    h, w = score_map.shape[:2]
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(poly_pts, dtype=np.float64).reshape(-1, 2)
    allp = np.vstack([np.round(q), p])
    x0 = int(np.clip(np.floor(allp[:, 0].min()), 0, w - 1)); x1 = int(np.clip(np.ceil(allp[:, 0].max()), 0, w - 1))
    y0 = int(np.clip(np.floor(allp[:, 1].min()), 0, h - 1)); y1 = int(np.clip(np.ceil(allp[:, 1].max()), 0, h - 1))
    sub = score_map[y0:y1 + 1, x0:x1 + 1]
    off = np.array([x0, y0], dtype=np.float64)
    qmask = np.zeros(sub.shape[:2], dtype=np.uint8)
    cv2.fillPoly(qmask, [(np.round(q) - off).astype(np.int32)], 1)
    pmask = np.zeros(sub.shape[:2], dtype=np.uint8)
    cv2.fillPoly(pmask, [(p.astype(np.int32) - off.astype(np.int32))], 1)
    ring = (qmask == 1) & (pmask == 0)
    quad_mean = float(sub[qmask == 1].mean()) if qmask.any() else 0.
    ring_mean = float(sub[ring].mean()) if ring.any() else quad_mean
    area_p = abs(cv2.contourArea(p.astype(np.float32)))
    area_q = abs(cv2.contourArea(q.astype(np.float32)))
    hull = abs(cv2.contourArea(cv2.convexHull(p.astype(np.float32))))
    (_, _), (rw, rh), _ = cv2.minAreaRect(q.astype(np.float32))
    short, long_ = min(rw, rh), max(rw, rh)
    return dict(ring_mean=ring_mean, quad_mean=quad_mean, fill_quad=area_p/area_q if area_q > 0 else 0.,
                fill_hull=area_p/hull if hull > 0 else 0., short_norm=short/max(tensor_short, 1),
                log_aspect=math.log(max(long_, 1e-6)/max(short, 1e-6)), n_vertices=float(len(p)))


def feature_vector(stats, qf):
    d = dict(raw=stats['mean'], std=stats['std'], q10=stats['q10'], q50=stats['q50'], q90=stats['q90'], pmax=stats['pmax'], **qf)
    return [float(d[k]) for k in FEATURES]


# ---------------------------------------------------------------- instrumented post-processing (GPU path)

def capture_with_features(native, probability, sample, restore, geometry=GEOM):
    """Tuan's installed DB post-processor with min_text_score = 0, hooking _get_bbox_score/_unclip exactly like
    rev4.decision_core.capture_candidates; for each ACCEPTED candidate also returns the feature vector."""
    import cv2
    import numpy as np
    from lowres.inference import unpack
    name, mask, ratio, width, epsilon = geometry
    processor = copy.copy(native)
    processor.mask_thr, processor.unclip_ratio = mask, ratio
    processor.min_text_width, processor.epsilon_ratio = width, epsilon
    processor.min_text_score = 0.
    require(processor.text_repr_type == 'quad', 'Only the frozen quad path is supported')
    score_method, unclip_method = processor._get_bbox_score, processor._unclip
    tensor_short = float(min(sample.metainfo['img_shape'][:2]))
    events = []

    def observe_score(score_map, polygon):
        score = float(score_method(score_map, polygon))
        st = region_stats(score_map, polygon)
        require(abs(st['mean'] - score) < 1e-9, 'Region mean differs from the native score')
        events.append(dict(raw=score, status='scored', poly=np.asarray(polygon).copy(), map=score_map, stats=st))
        return score

    def observe_unclip(polygon):
        result = unclip_method(polygon)
        e = events[-1]
        if len(result) == 0:
            e['status'] = 'unclip_empty'
        else:
            rect = cv2.minAreaRect(result.reshape(-1, 2))
            e['status'] = 'accepted' if min(rect[1]) >= width else 'width_rejected'
            e['quad'] = cv2.boxPoints(rect)
        return result

    processor._get_bbox_score, processor._unclip = observe_score, observe_unclip
    prediction = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
    acc = [e for e in events if e['status'] == 'accepted']
    require(len(acc) == len(prediction['polygons']), 'Observed native output count changed')
    feats = [feature_vector(e['stats'], quad_features(e['map'], e['poly'], e['quad'], tensor_short)) for e in acc]
    return prediction, [e['raw'] for e in acc], feats


# ---------------------------------------------------------------- prepare / extract / devcheck (GPU side)

def prepare(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from PIL import Image
    from rev4.training_data import native_lr2
    rows = load_rows(a.rows)
    made = bad = 0
    for t in tasks_for(rows, set(a.roles.split(',')), a.clean_root, a.prepared_root):
        if t['condition'] == 'clean':
            require(Path(t['path']).is_file() and sha256_file(t['path']) == t['sha256'], f'Clean bytes differ: {t["path"]}')
            with Image.open(t['path']) as im:
                require(list(im.size) == t['size'], f'Clean size differs from TextOCR for {t["source"]}: {im.size} vs {t["size"]}')
            continue
        out = Path(t['path'])
        if not out.is_file():
            with Image.open(Path(a.clean_root)/rows[t['source']]['clean_relpath']) as im:
                lr2 = native_lr2(im)
            lr2.info.clear()
            out.parent.mkdir(parents=True, exist_ok=True)
            lr2.save(out, format='PNG')
            made += 1
        with Image.open(out) as im:
            if list(im.size) != t['size']:
                bad += 1
    require(bad == 0, f'{bad} LR2 images have an unexpected size')
    print(json.dumps(dict(lr2_made=made)))


def _detector(a):
    from mixab.monkey_janek import use_monkey
    monkey = use_monkey()
    from rev4.stage_a import detector_for
    manifest = read_json(Path(a.weights_dir)/'merge_manifest.json')
    b1 = next(m for m in manifest['models'] if m['alpha'] == '1')
    ckpt = Path(a.weights_dir)/PureWindowsPath(b1['checkpoint']).name
    require(sha256_file(ckpt) == b1['checkpoint_sha256'], f'{ckpt} differs from merge manifest')
    det, native, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=ckpt))
    return monkey, det, native


def _forward(det, path):
    import torch
    _, inputs = next(iter(det.preprocess([str(path)], batch_size=1)))
    with torch.inference_mode():
        processed = det.model.data_preprocessor(inputs, training=False)
        prob = det.model(**processed, mode='tensor')
    return processed, prob


def run_task(det, native, task, consistency):
    import numpy as np
    import torch
    from rev4.decision_core import capture_candidates
    from rev4.decision_plan import GEOMETRIES
    W, H = task['size']
    clean_w, clean_h = (task['size'] if task['condition'] == 'clean' else task['clean_size'])
    restore = np.asarray([clean_w, clean_h])/np.asarray([W, H])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    processed, prob = _forward(det, task['path'])
    torch.cuda.synchronize(); t1 = time.perf_counter()
    sample = processed['data_samples'][0]
    pred, raw, _ = capture_candidates(native, prob, sample, restore, {g[0]: g for g in GEOMETRIES}[GEOM[0]])
    t2 = time.perf_counter()
    pred2, raw2, feats = capture_with_features(native, prob, sample, restore)
    t3 = time.perf_counter()
    same = None
    if consistency:
        same = (raw == raw2 and len(pred['polygons']) == len(pred2['polygons']) and
                all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(pred['polygons'], pred2['polygons'])))
        require(same, f'Instrumented post-processing differs from Tuan capture_candidates on {task["source"]}/{task["condition"]}')
    keep = [k for k, r in enumerate(raw2) if r >= FLOOR]
    feat_ok = all(all(math.isfinite(v) for v in f) for f in feats)
    require(feat_ok, f'Non-finite feature on {task["source"]}/{task["condition"]}')
    return dict(source=task['source'], condition=task['condition'], role=task.get('role'), size=[W, H],
                polygons=[pred2['polygons'][k] for k in keep], raw=[raw2[k] for k in keep],
                packed=[pred2['scores'][k] for k in keep], features=[feats[k] for k in keep],
                ms=dict(forward=(t1 - t0)*1000, postprocess_plain=(t2 - t1)*1000, postprocess_features=(t3 - t2)*1000),
                consistency=same, img_shape=[int(v) for v in sample.metainfo['img_shape'][:2]],
                image_sha256=sha256_file(task['path']))


def extract(a):
    import torch
    rows = load_rows(a.rows)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need exactly one visible GPU')
    monkey, det, native = _detector(a)
    tasks = tasks_for(rows, set(a.roles.split(',')), a.clean_root, a.prepared_root)
    for t in tasks:
        t['clean_size'] = rows[t['source']]['clean_size']
    done = set()
    for p in sorted(out.glob('shard_*.jsonl.gz')):
        done |= {(r['source'], r['condition']) for r in map(json.loads, gzip.open(p, 'rt', encoding='utf-8'))}
    todo = [t for t in tasks if (t['source'], t['condition']) not in done]
    print(f'{len(tasks)} tasks, {len(done)} done, {len(todo)} to run', flush=True)
    torch.cuda.reset_peak_memory_stats()
    buf, sid, tic = [], len(list(out.glob('shard_*.jsonl.gz'))), time.perf_counter()
    for n, t in enumerate(todo, 1):
        if t['sha256']:
            require(sha256_file(t['path']) == t['sha256'], f'Image SHA mismatch {t["path"]}')
        buf.append(run_task(det, native, t, a.consistency or n % 50 == 1))
        if len(buf) == a.shard or n == len(todo):
            path = out/f'shard_{sid:04d}.jsonl.gz'
            with gzip.open(path.with_suffix('.tmp'), 'wt', encoding='utf-8') as f:
                for r in buf:
                    f.write(json.dumps(r) + '\n')
            path.with_suffix('.tmp').replace(path)
            sid += 1; buf = []
            el = time.perf_counter() - tic
            print(f'{n}/{len(todo)} · {el/n:.2f} s/task · ETA {(len(todo) - n)*el/n/60:.0f} min', flush=True)
    recs = load_records(out)
    ms = {k: [r['ms'][k] for r in recs] for k in ('forward', 'postprocess_plain', 'postprocess_features')}
    import numpy as np
    summary = dict(monkey_v2=monkey, gpu=torch.cuda.get_device_name(0), tasks=len(tasks), ran=len(todo),
                   seconds_per_task_wall=(time.perf_counter() - tic)/max(len(todo), 1),
                   peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
                   consistency_checked=sum(r['consistency'] is not None for r in recs),
                   consistency_all_true=all(r['consistency'] is not False for r in recs),
                   candidates=sum(len(r['raw']) for r in recs),
                   ms={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in ms.items()},
                   feature_overhead_ratio_median=float(np.median([r['ms']['postprocess_features']/max(r['ms']['postprocess_plain'], 1e-6) for r in recs])),
                   projected_minutes_for_4000_tasks=(time.perf_counter() - tic)/max(len(todo), 1)*4000/60)
    save_json(out/'extract_summary.json', summary)
    print(json.dumps(summary, indent=1))


def devcheck(a):
    """Pipeline check on a few development LR2 images (no labels read, nothing fitted): candidates of the plain path
    must equal the T4 dump (B1, mask_040) and the instrumented path must equal the plain path."""
    from mixab.monkey_janek import use_monkey
    use_monkey()  # This CLI runs in a fresh process; initialize before importing rev4.
    import numpy as np
    from mixab import dualagree_janek as da 
    from mixab import merge_audit_janek as ma 
    rows = ma.load_rows(a.bundle)
    meta = [m for m in ma.load_task_metadata(a.bundle) if m['condition'] == 'lr2']
    srcs = json.loads(Path(a.sources).read_text(encoding='utf-8'))['sources']
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    tasks = [t for t in ma.resolve_tasks(meta, roots, srcs) if t['condition'] == 'lr2']
    from PIL import Image
    from rev4.training_data import native_lr2
    for t in tasks:
        out = Path(t['path'])
        if not out.is_file():
            with Image.open(Path(a.clean_root)/rows[t['source_image_id']]['clean_relpath']) as im:
                lr2 = native_lr2(im)
            lr2.info.clear(); out.parent.mkdir(parents=True, exist_ok=True); lr2.save(out, format='PNG')
        require(sha256_file(out) == t['image_sha256'], f'LR2 bytes differ from the locked task {out}')
    monkey, det, native = _detector(a)
    ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
    from rev4.decision_plan import GEOMETRIES
    dl = da.load_model_dump(Path(a.dump)/'B1', len(GEOMETRIES), ident)
    gi = [g[0] for g in GEOMETRIES].index(GEOM[0])
    res = []
    for t in tasks:
        r = run_task(det, native, dict(source=t['source_image_id'], condition='lr2', path=t['path'], size=list(t['size']),
                                       clean_size=rows[t['source_image_id']]['clean_size']), True)
        e = dl[t['sample_id']][gi]
        same = (len(e['raw']) == len(r['raw']) and np.allclose(e['raw'], r['raw'], atol=0, rtol=0) and
                all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(e['polygons'], r['polygons'])))
        res.append(dict(source=t['source_image_id'], dump_candidates=len(e['raw']), live_candidates=len(r['raw']), identical=bool(same),
                        max_abs_raw_diff=float(max((abs(x - y) for x, y in zip(e['raw'], r['raw'])), default=0.))))
    out = dict(sources=len(res), identical=sum(x['identical'] for x in res), detail=res,
               note='T4 vs T4 dump (Colab). Exact identity expected on the same GPU type; any drift is reported, not hidden.')
    save_json(Path(a.output)/'devcheck.json', out)
    print(json.dumps(out, indent=1))


def smokecheck(a):
    """CPU, smoke tasks only: official evaluator == CandidateMatcher at F .40/.60 on the full-canvas TRAIN rows, the stored
    raw feature == raw score, label time per task (projection for 4,000 tasks)."""
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from rev4.decision_core import CandidateMatcher
    from mixab import merge_audit_janek as ma 
    policy = read_json(Path(a.bundle)/'provenance/textvqa_evaluation_protocol.json')
    rows = load_rows(a.rows)
    recs = load_records(a.extract_dir)
    out, secs = [], []
    for r in recs:
        row = rows[r['source']]
        m = CandidateMatcher(dict(polygons=r['polygons'], scores=r['packed']), r['raw'], row, policy)
        fast = m.assert_official(CUT_F)               # raises if it differs from the unchanged evaluator
        t0 = time.perf_counter()
        lab, _ = task_labels(r, row, policy)
        secs.append(time.perf_counter() - t0)
        out.append(dict(source=r['source'], condition=r['condition'], candidates=len(r['raw']), counts_F=fast['counts'],
                        positives=sum(l_ == 1 for l_ in lab), negatives=sum(l_ == 0 for l_ in lab), dropped=sum(l_ is None for l_ in lab),
                        raw_feature_equals_raw=all(f[0] == v for f, v in zip(r['features'], r['raw']))))
    import numpy as np
    res = dict(tasks=len(out), official_evaluator_agrees=True, raw_feature_ok=all(x['raw_feature_equals_raw'] for x in out),
               label_seconds_per_task_median=float(np.median(secs)), detail=out,
               F_040_060_smoke=dict(tp=sum(x['counts_F'][0] for x in out), gt=sum(x['counts_F'][1] for x in out),
                                    pred=sum(x['counts_F'][2] for x in out)),
               note='Smoke TRAIN sources (not in fit/calibration). B1 trained on TRAIN, so these counts are optimistic by design.')
    save_json(Path(a.output)/'smokecheck.json', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'detail'}, indent=1))


def load_records(folder):
    recs = []
    for p in sorted(Path(folder).glob('shard_*.jsonl.gz')):
        recs += [json.loads(line) for line in gzip.open(p, 'rt', encoding='utf-8')]
    return recs


# ---------------------------------------------------------------- labels, model, calibration (CPU)

def task_labels(rec, row, policy):
    """Per candidate: 1 (IoU >= match_iou edge to a valid GT), 0 (counted, no edge), None (dropped by the evaluator's
    ROI/ignore filter: does not change counts). Also returns the matcher for counting."""
    from rev4.decision_core import CandidateMatcher
    m = CandidateMatcher(dict(polygons=rec['polygons'], scores=rec['packed']), rec['raw'], row, policy)
    edge = set().union(*m.edges) if m.edges else set()
    lab = []
    for p in range(len(rec['raw'])):
        if p in m.geometries:
            lab.append(1 if p in edge else 0)
        elif p in m.invalid:
            lab.append(0)
        else:
            lab.append(None)
    return lab, m


def standardize_fit(X, w):
    import numpy as np
    mu = (X*w[:, None]).sum(0)/w.sum()
    sd = np.sqrt(((X - mu)**2*w[:, None]).sum(0)/w.sum())
    sd[sd < 1e-12] = 1.
    return mu, sd


def fit_logistic(Z, y, w, l2=L2, iters=100):
    """Weighted L2 logistic regression by Newton/IRLS on standardized Z (intercept unpenalized). Deterministic."""
    import numpy as np
    n, d = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2); reg[0] = 0.
    wn = w/w.sum()
    for _ in range(iters):
        p = 1/(1 + np.exp(-np.clip(A @ beta, -40, 40)))
        g = A.T @ (wn*(p - y)) + reg*beta/n
        Hm = (A*(wn*p*(1 - p))[:, None]).T @ A + np.diag(reg/n)
        step = np.linalg.solve(Hm, g)
        beta -= step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def predict(beta, Z):
    import numpy as np
    return 1/(1 + np.exp(-np.clip(np.hstack([np.ones((len(Z), 1)), Z]) @ beta, -40, 40)))


def hmean(t):
    return 2*t[0]/(t[1] + t[2]) if t[1] + t[2] else 0.


def counts_at(m, scores, cut, groups):
    """[tp, gt, pred, gt<4, tp<4] for the kept set {score >= cut} with the official maximum matching."""
    from mixab import phase0_janek as p0 
    kept = {i for i, s in enumerate(scores) if s >= cut}
    pairs, inv = p0.match_pairs(m, kept)
    row = p0.row_from_pairs(m, groups, pairs, inv, kept)
    return [row[0], row[1], row[2], row[3], row[4]]


def mean_h(rows_by_cond):
    """mean of Clean and LR2 Hmean, each from SUMMED counts."""
    return (hmean(rows_by_cond['clean'].sum(0)) + hmean(rows_by_cond['lr2'].sum(0)))/2


def choose_cut(per_cut, tiny_ref):
    """per_cut: list of (cut, rows_by_cond). Max mean Hmean s.t. LR2 tiny >= tiny_ref + TINY_PP; ties -> higher cut."""
    best = None
    for cut, rb in per_cut:
        t = rb['lr2'].sum(0)
        tiny = t[4]/t[3] if t[3] else 0.
        if tiny*100 < tiny_ref*100 + TINY_PP:
            continue
        key = (round(mean_h(rb), 12), cut)
        if best is None or key > best[0]:
            best = (key, cut, rb)
    return best


def paired_boot_mean(a, b, seed=BOOT_SEED, n=BOOT_N):
    """a/b: rows_by_cond with the SAME source order; resample sources (Clean+LR2 pair kept) -> Δ mean Hmean (pp)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    k = len(a['lr2'])
    d = []
    for _ in range(n):
        j = rng.integers(0, k, k)
        d.append((mean_h({c: a[c][j] for c in a}) - mean_h({c: b[c][j] for c in b}))*100)
    return dict(delta_pp=(mean_h(a) - mean_h(b))*100, ci95_pp=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], sources=k)


def label_outcome(delta, ci_low, score_only_matches_raw):
    if not score_only_matches_raw:
        return 'INVALID_CONTROL'
    if delta <= 0:
        return 'NO_SIGNAL'
    if ci_low <= 0:
        return 'INCONCLUSIVE'
    return 'PASS_CALIBRATION_SCREEN' if delta >= GATE_PP else 'POSITIVE_BELOW_GATE'


def fit_cmd(a):
    import numpy as np
    from mixab.monkey_janek import use_monkey
    use_monkey()                                   # rev4 on sys.path; NO development rows are loaded here
    policy = read_json(Path(a.bundle)/'provenance/textvqa_evaluation_protocol.json')
    rows = load_rows(a.rows)
    recs = load_records(a.extract_dir)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'boxquality_fit.json').exists(), f'Refusing to overwrite {out}/boxquality_fit.json')
    by = {(r['source'], r['condition']): r for r in recs}
    roles = {s: rows[s]['role'] for s in rows}
    need = [(s, c) for s in sorted(rows) if roles[s] in ('fit', 'calibration') for c in ('clean', 'lr2')]
    missing = [k for k in need if k not in by]
    require(not missing, f'{len(missing)} fit/calibration tasks missing in the extraction, e.g. {missing[:3]}')
    X, y, w, info = [], [], [], []
    mats = {}
    tic = time.perf_counter()
    for s, c in need:
        r = by[(s, c)]
        lab, m = task_labels(r, rows[s], policy)
        mats[(s, c)] = m
        used = [i for i, l_ in enumerate(lab) if l_ is not None]
        for i in used:
            X.append(r['features'][i]); y.append(lab[i]); w.append(.5/len(used)); info.append((s, c, i))
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    fold = np.array([roles[s] for s, _, _ in info])
    print(f'labels: {len(y)} candidates ({time.perf_counter() - tic:.0f} s)', flush=True)
    fit = fold == 'fit'
    mu, sd = standardize_fit(X[fit], w[fit])
    Z = (X - mu)/sd
    beta_full = fit_logistic(Z[fit], y[fit], w[fit])
    beta_raw = fit_logistic(Z[fit][:, :1], y[fit], w[fit])
    score = dict(RAW={}, SCORE_ONLY={}, FULL={})
    p_full, p_raw = predict(beta_full, Z), predict(beta_raw, Z[:, :1])
    for k, (s, c, i) in enumerate(info):
        for name, val in (('RAW', X[k, 0]), ('SCORE_ONLY', p_raw[k]), ('FULL', p_full[k])):
            score[name].setdefault((s, c), {})[i] = float(val)
    # calibration: every candidate of a task gets a score (dropped ones do not affect counts)
    cal_src = sorted(s for s in rows if roles[s] == 'calibration')
    groups_of = {}

    def rows_for(name, cut):
        rb = {}
        for c in ('clean', 'lr2'):
            arr = []
            for s in cal_src:
                m = mats[(s, c)]
                if (s, c) not in groups_of:
                    groups_of[(s, c)] = [{i for i, h in enumerate(m.heights) if lo <= h < hi} for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
                sc = score[name].get((s, c), {})
                vec = [sc.get(i, -1.) for i in range(len(by[(s, c)]['raw']))]
                arr.append(counts_at(m, vec, cut, groups_of[(s, c)]))
            rb[c] = np.asarray(arr, dtype=np.int64)
        return rb
    F = {}
    for c in ('clean', 'lr2'):
        F[c] = np.asarray([counts_at(mats[(s, c)], by[(s, c)]['raw'], CUT_F, [{i for i, h in enumerate(mats[(s, c)].heights) if lo <= h < hi}
                           for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]) for s in cal_src], dtype=np.int64)
    tF = F['lr2'].sum(0)
    tiny_ref = tF[4]/tF[3] if tF[3] else 0.
    chosen = {}
    for name in ('RAW', 'SCORE_ONLY', 'FULL'):
        vals = np.asarray([v for (s, c), d in score[name].items() if roles[s] == 'calibration' for v in d.values()])
        cuts = sorted(set(float(x) for x in np.quantile(vals, np.linspace(0, 1, N_CUTS))))
        per = []
        for k, cut in enumerate(cuts):
            per.append((cut, rows_for(name, cut)))
            if k % 20 == 0:
                print(f'{name}: cut {k + 1}/{len(cuts)} ({time.perf_counter() - tic:.0f} s)', flush=True)
        best = choose_cut(per, tiny_ref)
        require(best is not None, f'No cutoff satisfies the tiny constraint for {name}')
        chosen[name] = dict(cut=best[1], rows=best[2], curve=[dict(cut=cu, mean_hmean=mean_h(rb),
                                                              lr2=hmean(rb['lr2'].sum(0)), clean=hmean(rb['clean'].sum(0))) for cu, rb in per])

    def summary(rb):
        out_ = {}
        for c in ('clean', 'lr2'):
            t = rb[c].sum(0)
            out_[c] = dict(tp=int(t[0]), gt=int(t[1]), pred=int(t[2]), fp=int(t[2] - t[0]), hmean=hmean(t),
                           tiny_recall=float(t[4]/t[3]) if t[3] else None)
        out_['mean_hmean'] = mean_h(rb)
        return out_
    full_vs_raw = paired_boot_mean(chosen['FULL']['rows'], chosen['RAW']['rows'])
    so_vs_raw = paired_boot_mean(chosen['SCORE_ONLY']['rows'], chosen['RAW']['rows'])
    same = abs(so_vs_raw['delta_pp']) < .05
    res = dict(label='Box-quality scorer: fit on TRAIN-fit, cutoff on TRAIN-calibration (both B1 TRAIN; not independent of the '
                     'detector). NOT a development result, NOT an improvement claim.', spec='docs/research/janek/boxquality_spec_20261124_janek.md',
               candidates=dict(total=int(len(y)), fit=int(fit.sum()), calibration=int((fold == 'calibration').sum()),
                               positive_rate_fit=float((y[fit]*w[fit]).sum()/w[fit].sum())),
               features=list(FEATURES), l2=L2, standardizer=dict(mean=mu.tolist(), std=sd.tolist()),
               beta_full=beta_full.tolist(), beta_score_only=beta_raw.tolist(),
               F_040_060_calibration=summary(F), tiny_ref_lr2=float(tiny_ref),
               chosen={k: dict(cut=v['cut'], **summary(v['rows'])) for k, v in chosen.items()},
               full_minus_raw=full_vs_raw, score_only_minus_raw=so_vs_raw, full_minus_F=paired_boot_mean(chosen['FULL']['rows'], F),
               raw_minus_F=paired_boot_mean(chosen['RAW']['rows'], F),
               score_only_matches_raw=bool(same),
               outcome=label_outcome(full_vs_raw['delta_pp'], full_vs_raw['ci95_pp'][0], same))
    save_json(out/'boxquality_fit.json', res)
    save_json(out/'curves.json', {k: v['curve'] for k, v in chosen.items()})
    np.savez_compressed(out/'model.npz', mean=mu, std=sd, beta_full=beta_full, beta_score_only=beta_raw,
                        features=np.asarray(FEATURES), cut_full=chosen['FULL']['cut'], cut_raw=chosen['RAW']['cut'])
    review(a, rows, by, info, y, p_full, fold, out)
    print(json.dumps({k: res[k] for k in ('candidates', 'F_040_060_calibration', 'chosen', 'full_minus_raw', 'score_only_minus_raw',
                                          'full_minus_F', 'raw_minus_F', 'outcome')}, indent=1, default=str))


def review(a, rows, by, info, y, p_full, fold, out):
    """Hash-selected review sheet (calibration only): 24 counted negatives with NO overlap to any valid GT and the
    highest FULL scores' top decile (possible missing labels), + 8 positives. Descriptive; labels are not changed."""
    import numpy as np
    from PIL import Image, ImageDraw
    from shapely.geometry import shape
    if not a.clean_root:
        return
    from mixab import dualagree_janek as da 
    cal = np.where(fold == 'calibration')[0]
    thr = np.quantile(p_full[cal], .9)
    neg, pos = [], []
    for k in cal:
        s, c, i = info[k]
        q = da.polygon_of(by[(s, c)]['polygons'][i])
        gts = [shape(e['geometry']) for e in rows[s]['gt_clean']['valid']]
        touch = any(q is not None and q.intersects(g) for g in gts)
        rec = (s, c, i, float(p_full[k]))
        if y[k] == 0 and not touch and p_full[k] >= thr:
            neg.append(rec)
        elif y[k] == 1:
            pos.append(rec)
    rk = lambda r: _rank(SEED + 'review:' + f'{r[0]}:{r[1]}:{r[2]}')  # noqa: E731
    pick = [('neg_no_gt_high', r) for r in sorted(neg, key=rk)[:24]] + [('pos', r) for r in sorted(pos, key=rk)[:8]]
    folder = out/'review'
    folder.mkdir(exist_ok=True)
    sheet = []
    for n, (kind, (s, c, i, sc)) in enumerate(pick, 1):
        r = rows[s]
        with Image.open(Path(a.clean_root)/r['clean_relpath']) as im:
            img = im.convert('RGB')
        pts = np.asarray(by[(s, c)]['polygons'][i], float).reshape(-1, 2)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        mg = max(20., .5*max(x1 - x0, y1 - y0))
        crop = img.crop((int(max(0, x0 - mg)), int(max(0, y0 - mg)), int(min(img.width, x1 + mg)), int(min(img.height, y1 + mg))))
        d = ImageDraw.Draw(crop)
        off = np.array([max(0, x0 - mg), max(0, y0 - mg)])
        d.polygon([tuple(p) for p in (pts - off)], outline=(255, 120, 0), width=3)
        for e in r['gt_clean']['valid'] + r['gt_clean']['ignored']:
            g = np.asarray(shape(e['geometry']).exterior.coords) - off
            d.polygon([tuple(p) for p in g], outline=(0, 200, 0) if e in r['gt_clean']['valid'] else (150, 150, 150), width=1)
        path = folder/f'{n:02d}_{kind}.png'
        crop.save(path)
        sheet.append(dict(n=n, kind=kind, source=s, condition=c, candidate=i, full_score=sc, file=path.name,
                          reviewer_label=''))
    with (folder/'review_sheet.csv').open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, list(sheet[0]) if sheet else ['n'])
        wr.writeheader(); wr.writerows(sheet)
    save_json(folder/'README.json', dict(rule='calibration fold only; hash rank with seed ' + SEED + 'review:', negatives=len(neg),
                                         note='Ghi reviewer_label = text_missing_label | background | unsure. Mô tả; không đổi nhãn.'))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('rows')
    for arg in ('--textocr', '--train-ids', '--output'):
        r.add_argument(arg, required=True, type=Path)
    p = sub.add_parser('prepare')
    e = sub.add_parser('extract')
    for s in (p, e):
        s.add_argument('--rows', required=True, type=Path)
        s.add_argument('--clean-root', required=True, type=Path)
        s.add_argument('--prepared-root', required=True, type=Path)
        s.add_argument('--roles', default='smoke')
    for arg in ('--model-config', '--weights-dir', '--output'):
        e.add_argument(arg, required=True, type=Path)
    e.add_argument('--consistency', action='store_true')
    e.add_argument('--shard', type=int, default=50)
    d = sub.add_parser('devcheck')
    for arg in ('--bundle', '--sources', '--clean-root', '--prepared-root', '--model-config', '--weights-dir', '--dump', '--output'):
        d.add_argument(arg, required=True, type=Path)
    sc = sub.add_parser('smokecheck')
    for arg in ('--bundle', '--rows', '--extract-dir', '--output'):
        sc.add_argument(arg, required=True, type=Path)
    f = sub.add_parser('fit')
    for arg in ('--bundle', '--rows', '--extract-dir', '--output'):
        f.add_argument(arg, required=True, type=Path)
    f.add_argument('--clean-root', type=Path)
    a = ap.parse_args()
    if getattr(a, 'output', None) and a.command in ('devcheck', 'smokecheck', 'fit'):
        Path(a.output).mkdir(parents=True, exist_ok=True)
    dict(rows=rows_cmd, prepare=prepare, extract=extract, devcheck=devcheck, smokecheck=smokecheck, fit=fit_cmd)[a.command](a)


if __name__ == '__main__':
    main()
