"""Box-quality scorer · supplementary evaluation on development split A (Clean + LR2), frozen scorer.
Spec (written before running): docs/research/janek/boxquality_devA_spec_20261125_janek.md. A post-pilot user decision;
the pilot label POSITIVE_BELOW_GATE is unchanged. Nothing is fitted or re-selected here; B and TEST are never opened.

Commands
  prepare   A tasks: verify Clean bytes, build native LR2 (Tuan's native_lr2) and check the locked task SHA-256   CPU
  extract   B1 forward + instrumented post-processing on A tasks (mixab.boxquality_janek.run_task, consistency on
            EVERY task); --subset smoke|all; resumable shards                                                    GPU
  latency   fair end-to-end latency, F (B1 .40/.60 as deployed) vs FULL (features + frozen scorer), interleaved,
            plus peak GPU memory per arm                                                                         GPU
  evaluate  official evaluator for F / RAW / FULL on every task, fast path must agree, references from the T4 dumps,
            paired bootstrap, group gates, locked outcome label                                                  CPU
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

from mixab import read_json, require, save_json
from mixab import boxquality_janek as bq

SPEC = 'docs/research/janek/boxquality_devA_spec_20261125_janek.md'
MODEL_SHA256 = '81867a494e391db180d293635e63bfdd87ddac59b09b5aac9f18417159196eb9'
FIT_SHA256 = '76386825b11f9fff7d31e6129f1d7cdde67897ce7633aadf205eb6552d5ab8c3'
SPLIT = 'A'
SMOKE_SEED, N_SMOKE = '20261125:bq-devA:smoke:', 10
LAT_SEED, N_LAT, N_WARM = '20261125:bq-devA:latency:', 100, 5
CUT_F = bq.CUT_F                                  # B1 .40/.60
ARMS = ('F', 'RAW', 'FULL')
CONDS = ('clean', 'lr2')
BINS = ((0, 4), (4, 8), (8, 16), (16, float('inf')))
GATE_MEAN_PP, GATE_CLEAN_PP, GATE_TINY_PP = .5, -.3, -3.
USEFUL_TINY_PP, USEFUL_LATENCY = -3., 1.10
REF_SINGLES = {'B1 .40/.60': ('B1', 'mask_040', .6), 'B1 eps005/.50': ('B1', 'epsilon_005', .5),
               'B-Clean .40/.60': ('B-Clean', 'mask_040', .6)}


# ---------------------------------------------------------------- pure helpers (CPU-testable)

def _rank(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def a_sources(rows):
    from mixab import reselect_split_janek as rs
    return sorted(s for s in rows if rs.split_of(s) == SPLIT)


def smoke_sources(sources):
    return sorted(sorted(sources, key=lambda s: _rank(SMOKE_SEED + s))[:N_SMOKE])


def latency_sources(sources):
    """(warm-up sources, timed sources), disjoint, fixed by hash."""
    order = sorted(sources, key=lambda s: _rank(LAT_SEED + s))
    return sorted(order[:N_WARM]), sorted(order[N_WARM:N_WARM + N_LAT])


def load_model(model_path, fit_path):
    """Frozen scorer; refuses anything but the pilot's exact model and cutoffs."""
    import numpy as np
    require(bq.sha256_file(model_path) == MODEL_SHA256, f'{model_path} is not the frozen scorer')
    require(bq.sha256_file(fit_path) == FIT_SHA256, f'{fit_path} is not the frozen fit result')
    z = np.load(model_path)
    fit = read_json(fit_path)
    m = dict(mean=z['mean'], std=z['std'], beta=z['beta_full'], cut_full=float(z['cut_full']), cut_raw=float(z['cut_raw']),
             features=[str(x) for x in z['features']])
    require(m['features'] == list(bq.FEATURES), 'Feature list differs from the frozen FEATURES')
    require(m['cut_full'] == fit['chosen']['FULL']['cut'] and m['cut_raw'] == fit['chosen']['RAW']['cut'],
            'Cutoffs in model.npz differ from boxquality_fit.json')
    return m


def full_scores(model, features):
    """p_FULL for every candidate (same standardization and predict() as the fit)."""
    import numpy as np
    if not features:
        return []
    X = np.asarray(features, dtype=float)
    return [float(v) for v in bq.predict(model['beta'], (X - model['mean'])/model['std'])]


def kept_indices(raw, p_full, model):
    return dict(F=[i for i, r in enumerate(raw) if r >= CUT_F],
                RAW=[i for i, r in enumerate(raw) if r >= model['cut_raw']],
                FULL=[i for i, p in enumerate(p_full) if p >= model['cut_full']])


def bin_row(tp, gt, pred, matched_ids, heights):
    """11-column row [tp, gt, pred, (gt_bin, tp_bin) x 4] from official matched ann ids and LR2-equivalent heights."""
    row = [int(tp), int(gt), int(pred)]
    matched = set(matched_ids)
    for lo, hi in BINS:
        ids = [a for a, h in heights.items() if lo <= h < hi]
        row += [len(ids), sum(a in matched for a in ids)]
    return row


def totals(arr):
    """Summed 11-column counts -> rates."""
    import numpy as np
    t = np.asarray(arr, dtype=np.int64).sum(0)
    tp, gt, pred = int(t[0]), int(t[1]), int(t[2])
    out = dict(tp=tp, fp=pred - tp, fn=gt - tp, gt=gt, pred=pred, precision=tp/pred if pred else 0.,
               recall=tp/gt if gt else 0., hmean=bq.hmean(t))
    out['bins'] = {f'{lo}-{hi}': dict(gt=int(t[3 + 2*k]), tp=int(t[4 + 2*k]),
                                      recall=int(t[4 + 2*k])/int(t[3 + 2*k]) if int(t[3 + 2*k]) else None)
                   for k, (lo, hi) in enumerate(BINS)}
    out['tiny_recall'] = out['bins'][f'{BINS[0][0]}-{BINS[0][1]}']['recall']
    return out


def outcome(valid, d, ci, ci_clean_hi, ci_lr2_hi, tiny_clean_pp, tiny_lr2_pp, latency_ratio, gates):
    """Locked label (spec §5) from FULL - F on mean Hmean. Returns (label, failed SMALL_USEFUL conditions)."""
    if not valid:
        return 'INVALID_RUN', []
    if all(gates.values()) and ci[0] > 0:
        return 'GROUP_GATE_PASS', []
    if d <= 0:
        return 'NO_BENEFIT', []
    if ci[0] <= 0:
        return 'INCONCLUSIVE', []
    failed = []
    if ci_clean_hi < 0:
        failed.append('clean_significantly_worse')
    if ci_lr2_hi < 0:
        failed.append('lr2_significantly_worse')
    if tiny_clean_pp < USEFUL_TINY_PP:
        failed.append('tiny_clean_below_F_minus_3pp')
    if tiny_lr2_pp < USEFUL_TINY_PP:
        failed.append('tiny_lr2_below_F_minus_3pp')
    if latency_ratio is None or latency_ratio > USEFUL_LATENCY:
        failed.append('latency_median_ratio_above_1.10' if latency_ratio is not None else 'latency_missing')
    return ('POSITIVE_NOT_USEFUL', failed) if failed else ('SMALL_USEFUL', [])


def generalization(d, ci):
    if d <= 0:
        return 'khong'
    return 'tong_quat_hoa' if ci[0] > 0 else 'mot_phan'


def _plain(x):
    """numpy scalars -> Python (for JSON)."""
    if hasattr(x, 'item'):
        return x.item()
    raise TypeError(f'Not JSON serializable: {type(x).__name__}')


def same_candidates(a_polys, a_raw, b_polys, b_raw):
    import numpy as np
    return (len(a_raw) == len(b_raw) and all(x == y for x, y in zip(a_raw, b_raw)) and
            all(np.array_equal(np.asarray(x, float), np.asarray(y, float)) for x, y in zip(a_polys, b_polys)))


# ---------------------------------------------------------------- tasks (shared by GPU commands)

def _context(bundle):
    from mixab import merge_audit_janek as ma
    rows = ma.load_rows(bundle)
    return rows, a_sources(rows), ma


def dev_tasks(bundle, clean_root, prepared_root, sources):
    rows, _, ma = _context(bundle)
    roots = dict(TextVQA_clean_root=clean_root, prepared_root=prepared_root)
    tasks = ma.resolve_tasks(ma.load_task_metadata(bundle), roots, sources)
    for t in tasks:
        t['source'] = t['source_image_id']
        t['clean_size'] = rows[t['source']]['clean_size']
        t['role'] = 'devA'
    require(len(tasks) == 2*len(sources), f'Expected {2*len(sources)} tasks, got {len(tasks)}')
    return sorted(tasks, key=lambda t: (t['source'], t['condition']))


def pick(rows_sources, subset):
    return smoke_sources(rows_sources) if subset == 'smoke' else rows_sources


# ---------------------------------------------------------------- prepare (CPU)

def prepare(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from PIL import Image
    from rev4.training_data import native_lr2
    rows, A, _ = _context(a.bundle)
    tasks = dev_tasks(a.bundle, a.clean_root, a.prepared_root, pick(A, a.subset))
    made = 0
    for t in tasks:
        if t['condition'] == 'clean':
            require(Path(t['path']).is_file() and bq.sha256_file(t['path']) == t['image_sha256'], f'Clean bytes differ: {t["path"]}')
            continue
        out = Path(t['path'])
        if not out.is_file():
            with Image.open(Path(a.clean_root)/rows[t['source']]['clean_relpath']) as im:
                lr2 = native_lr2(im)
            lr2.info.clear()
            out.parent.mkdir(parents=True, exist_ok=True)
            lr2.save(out, format='PNG')
            made += 1
        require(bq.sha256_file(out) == t['image_sha256'], f'LR2 bytes differ from the locked task: {out}')
    print(json.dumps(dict(subset=a.subset, sources=len(tasks)//2, tasks=len(tasks), lr2_made=made, all_sha_ok=True)))


# ---------------------------------------------------------------- extract (GPU)

def extract(a):
    import numpy as np
    import torch
    _, A, _ = _context(a.bundle)
    tasks = dev_tasks(a.bundle, a.clean_root, a.prepared_root, pick(A, a.subset))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need exactly one visible GPU')
    monkey, det, native = bq._detector(a)
    done = {(r['source'], r['condition']) for r in bq.load_records(out)}
    todo = [t for t in tasks if (t['source'], t['condition']) not in done]
    print(f'{len(tasks)} tasks, {len(done)} done, {len(todo)} to run', flush=True)
    torch.cuda.reset_peak_memory_stats()
    buf, sid, tic = [], len(list(out.glob('shard_*.jsonl.gz'))), time.perf_counter()
    for n, t in enumerate(todo, 1):
        require(bq.sha256_file(t['path']) == t['image_sha256'], f'Image SHA mismatch {t["path"]}')
        r = bq.run_task(det, native, t, True)                       # consistency on EVERY task (spec §6.3)
        r['sample_id'] = t['sample_id']
        buf.append(r)
        if len(buf) == a.shard or n == len(todo):
            path = out/f'shard_{sid:04d}.jsonl.gz'
            with gzip.open(path.with_suffix('.tmp'), 'wt', encoding='utf-8') as f:
                for x in buf:
                    f.write(json.dumps(x) + '\n')
            path.with_suffix('.tmp').replace(path)
            sid += 1
            buf = []
            el = time.perf_counter() - tic
            print(f'{n}/{len(todo)} · {el/n:.2f} s/task · ETA {(len(todo) - n)*el/n/60:.0f} min', flush=True)
    recs = bq.load_records(out)
    ms = {k: [r['ms'][k] for r in recs] for k in ('forward', 'postprocess_plain', 'postprocess_features')}
    summary = dict(monkey_v2=monkey, gpu=torch.cuda.get_device_name(0), subset=a.subset, tasks=len(tasks), records=len(recs),
                   ran=len(todo), seconds_per_task_wall=(time.perf_counter() - tic)/max(len(todo), 1),
                   peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
                   consistency_checked=sum(r['consistency'] is not None for r in recs),
                   consistency_all_true=all(r['consistency'] is True for r in recs),
                   candidates=sum(len(r['raw']) for r in recs),
                   ms={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in ms.items()},
                   projected_minutes_for_all_A=(time.perf_counter() - tic)/max(len(todo), 1)*2*len(A)/60)
    save_json(out/'extract_summary.json', summary)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- latency (GPU)

def _quant(v):
    import numpy as np
    v = np.asarray(v, float)
    return dict(n=int(v.size), median=float(np.median(v)), p95=float(np.percentile(v, 95)), mean=float(v.mean()),
                p10=float(np.percentile(v, 10)), max=float(v.max()))


def _paired(f_ms, full_ms, seed=bq.BOOT_SEED, n=bq.BOOT_N):
    import numpy as np
    f, g = np.asarray(f_ms, float), np.asarray(full_ms, float)
    rng = np.random.default_rng(seed)
    rm, rp, md = [], [], []
    for _ in range(n):
        j = rng.integers(0, f.size, f.size)
        rm.append(np.median(g[j])/np.median(f[j])); rp.append(np.percentile(g[j], 95)/np.percentile(f[j], 95))
        md.append(float((g[j] - f[j]).mean()))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return dict(pairs=int(f.size), ratio_of_medians=float(np.median(g)/np.median(f)), ratio_of_medians_ci95=ci(rm),
                ratio_of_p95=float(np.percentile(g, 95)/np.percentile(f, 95)), ratio_of_p95_ci95=ci(rp),
                mean_extra_ms=float((g - f).mean()), mean_extra_ms_ci95=ci(md))


def latency(a):
    import resource
    import numpy as np
    import torch
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from lowres.inference import unpack
    from mixab import csb1_latency_janek as cl
    out = Path(a.output)
    require(not (out/'latency.json').exists(), f'Refusing to overwrite {out}/latency.json')
    out.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need exactly one visible GPU')
    model = load_model(a.model, a.fit)
    rows, A, _ = _context(a.bundle)
    warm, timed = latency_sources(A)
    tasks = dev_tasks(a.bundle, a.clean_root, a.prepared_root, warm + timed)
    for t in tasks:                                               # SHA check + page-cache warm, outside timing
        require(bq.sha256_file(t['path']) == t['image_sha256'], f'Image SHA mismatch {t["path"]}')
        Path(t['path']).read_bytes()
    by_key = {(t['source'], t['condition']): t for t in tasks}
    ref = {(r['source'], r['condition']): r for r in bq.load_records(a.extract_dir)} if a.extract_dir else {}
    monkey, det, native = bq._detector(a)
    pp_f = copy.copy(native)
    _, mask, ratio, width, eps = bq.GEOM
    pp_f.mask_thr, pp_f.unclip_ratio, pp_f.min_text_width, pp_f.epsilon_ratio, pp_f.min_text_score = mask, ratio, width, eps, CUT_F
    sync = torch.cuda.synchronize

    def decode(path):
        _, inputs = next(iter(det.preprocess([str(path)], batch_size=1)))
        return inputs

    def fwd(inputs):
        with torch.inference_mode():
            processed = det.model.data_preprocessor(inputs, training=False)
            prob = det.model(**processed, mode='tensor')
        return processed['data_samples'][0], prob

    def restore_of(t):
        return np.asarray(t['clean_size'])/np.asarray(t['size'])

    def arm_f(t):
        s = [time.perf_counter()]
        inputs = decode(t['path']); s.append(time.perf_counter())
        sample, prob = fwd(inputs); sync(); s.append(time.perf_counter())
        polys = unpack(pp_f(prob, [copy.deepcopy(sample)])[0], restore_of(t))['polygons']; s.append(time.perf_counter())
        return polys, dict(pre=s[1] - s[0], fwd=s[2] - s[1], pp=s[3] - s[2], score=0.)

    def arm_full(t):
        s = [time.perf_counter()]
        inputs = decode(t['path']); s.append(time.perf_counter())
        sample, prob = fwd(inputs); sync(); s.append(time.perf_counter())
        pred, raw, feats = bq.capture_with_features(native, prob, sample, restore_of(t)); s.append(time.perf_counter())
        keep = [k for k, r in enumerate(raw) if r >= bq.FLOOR]
        p = full_scores(model, [feats[k] for k in keep])
        polys = [pred['polygons'][keep[j]] for j, v in enumerate(p) if v >= model['cut_full']]; s.append(time.perf_counter())
        return polys, dict(pre=s[1] - s[0], fwd=s[2] - s[1], pp=s[3] - s[2], score=s[4] - s[3])

    arms = dict(F=arm_f, FULL=arm_full)
    memory = {}
    for name, fn in arms.items():                                 # peak memory per arm on the warm-up images
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for s_ in warm:
            for c in CONDS:
                fn(by_key[(s_, c)])
        sync()
        memory[name] = dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, nvidia_smi=cl.gpu_state())
    timed_tasks = [by_key[(s_, c)] for s_ in timed for c in CONDS]
    samples, live = [], {}
    tic = time.perf_counter()
    for pass_index in range(a.passes):
        for i, t in enumerate(timed_tasks):
            first_f = (i % 2 == 0) != (pass_index % 2 == 1)
            for name in (('F', 'FULL') if first_f else ('FULL', 'F')):
                sync(); t0 = time.perf_counter()
                polys, st = arms[name](t)
                sync(); total = time.perf_counter() - t0
                samples.append(dict(pass_index=pass_index, source=t['source'], condition=t['condition'], arm=name,
                                    total_ms=total*1000, **{k: v*1000 for k, v in st.items()}))
                if pass_index == 0:
                    live[(t['source'], t['condition'], name)] = polys
        print(f'pass {pass_index + 1}/{a.passes} ({time.perf_counter() - tic:.0f} s)', flush=True)
    check = dict(images=len(timed_tasks), F_identical=0, FULL_identical=0, compared=0)
    for t in timed_tasks:
        r = ref.get((t['source'], t['condition']))
        if r is None:
            continue
        check['compared'] += 1
        p = full_scores(model, r['features'])
        want = dict(F=[q for q, v in zip(r['polygons'], r['raw']) if v >= CUT_F],
                    FULL=[q for q, v in zip(r['polygons'], p) if v >= model['cut_full']])
        for name in arms:
            got = live[(t['source'], t['condition'], name)]
            ok = len(got) == len(want[name]) and all(np.array_equal(np.asarray(x, float), np.asarray(y, float)) for x, y in zip(got, want[name]))
            check[f'{name}_identical'] += bool(ok)
    key = lambda x: (x['pass_index'], x['source'], x['condition'])  # noqa: E731
    by = {name: sorted([x for x in samples if x['arm'] == name], key=key) for name in arms}
    require([key(x) for x in by['F']] == [key(x) for x in by['FULL']], 'Unpaired latency samples')
    stages = ('pre', 'fwd', 'pp', 'score')
    res = dict(stage='box-quality dev A latency v1', spec=SPEC, monkey_v2=monkey, environment=cl.environment(torch),
               protocol=dict(batch=1, precision='fp32', passes=a.passes, timed_images=len(timed_tasks),
                             warmup_images=2*len(warm), seed=LAT_SEED, order='alternating per image, mirrored on pass 2',
                             F='decode -> B1 forward -> DB post-process mask_040, min_text_score .60 (as deployed)',
                             FULL='decode -> B1 forward -> capture_with_features (min_text_score 0, 13 features) -> '
                                  'raw >= .20 -> standardize + logistic -> p >= cut_full (implementation as frozen, unoptimized)',
                             excluded_from_timing=['image SHA-256', 'file read (page cache warmed)', 'model loading']),
               ms={name: dict(total=_quant([x['total_ms'] for x in by[name]]),
                              stages={st: _quant([x[st] for x in by[name]]) for st in stages},
                              by_condition={c: _quant([x['total_ms'] for x in by[name] if x['condition'] == c]) for c in CONDS})
                   for name in arms},
               paired=_paired([x['total_ms'] for x in by['F']], [x['total_ms'] for x in by['FULL']]),
               paired_by_condition={c: _paired([x['total_ms'] for x in by['F'] if x['condition'] == c],
                                               [x['total_ms'] for x in by['FULL'] if x['condition'] == c]) for c in CONDS},
               memory=memory, process_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               output_check=check, python=platform.python_version(),
               note='Comparable only inside this run. RSS is for the whole process (both arms, model, data).')
    with (out/'samples.jsonl').open('w', encoding='utf-8') as f:
        for x in samples:
            f.write(json.dumps(x) + '\n')
    save_json(out/'latency.json', res)
    print(json.dumps(dict(ms={k: v['total'] for k, v in res['ms'].items()}, paired=res['paired'], memory=memory,
                          check=check), indent=1))


# ---------------------------------------------------------------- evaluate (CPU)

def _ref_rows(dump_dir, cond, sources, rows, policy, geoms):
    """{single name: [11-col row per source]} + {source: B1 mask_040 entry} from an existing T4 dump."""
    from rev4.decision_core import CandidateMatcher
    from mixab import dualagree_janek as da
    from mixab import merge_audit_janek as ma
    ident = read_json(Path(dump_dir)/'dump_identity.json')
    require(ident['condition'] == cond, f'{dump_dir} is a {ident["condition"]} dump, expected {cond}')
    dumps = {name: da.load_model_dump(Path(dump_dir)/name, len(geoms), ident['identity_sha256']) for name in ('B1', 'B-Clean')}
    out = {k: [] for k in REF_SINGLES}
    b1_mask = {}
    for s in sources:
        sid = f'{cond}::{s}'
        for k, (model, geom, cut) in REF_SINGLES.items():
            e = dumps[model][sid][geoms.index(geom)]
            out[k].append(ma.count_row(CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], rows[s], policy).score(cut)))
        b1_mask[s] = dumps['B1'][sid][geoms.index(bq.GEOM[0])]
    return out, b1_mask, dict(gpu=ident.get('gpu'), identity=ident['identity_sha256'])


def evaluate(a):
    import numpy as np
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from lowres.evaluate_shared import evaluate_image
    from rev4.decision_core import CandidateMatcher
    from rev4.decision_plan import GEOMETRIES
    from mixab import phase0_janek as p0
    geoms = [g[0] for g in GEOMETRIES]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'devA_result.json').exists(), f'Refusing to overwrite {out}/devA_result.json')
    model = load_model(a.model, a.fit)
    rows, A, _ = _context(a.bundle)
    sources = pick(A, a.subset)
    policy = read_json(Path(a.bundle)/'provenance/textvqa_evaluation_protocol.json')
    recs = {(r['source'], r['condition']): r for r in bq.load_records(a.extract_dir)}
    missing = [(s, c) for s in sources for c in CONDS if (s, c) not in recs]
    require(not missing, f'{len(missing)} A tasks missing in the extraction (not silently dropped), e.g. {missing[:3]}')
    counts = {arm: {c: [] for c in CONDS} for arm in ARMS}
    checks = dict(tasks=0, consistency_all_true=True, fast_equals_official=True, nonfinite_features=0, fast_mismatch=[],
                  tiny_from_alternative_matching=0)
    kept_n = {arm: 0 for arm in ARMS}
    tic = time.perf_counter()
    for k, s in enumerate(sources):
        row = rows[s]
        for c in CONDS:
            r = recs[(s, c)]
            checks['tasks'] += 1
            checks['consistency_all_true'] &= r['consistency'] is True
            checks['nonfinite_features'] += sum(not all(math.isfinite(v) for v in f) for f in r['features'])
            p_full = full_scores(model, r['features'])
            m = CandidateMatcher(dict(polygons=r['polygons'], scores=r['packed']), r['raw'], row, policy)
            heights = dict(zip(m.ann_ids, m.heights.tolist()))
            groups = [{i for i, h in enumerate(m.heights) if lo <= h < hi} for lo, hi in BINS]
            for arm, idx in kept_indices(r['raw'], p_full, model).items():
                kept_n[arm] += len(idx)
                pred = dict(polygons=[r['polygons'][i] for i in idx], scores=[r['packed'][i] for i in idx])
                off = evaluate_image(row['gt_clean'], row['roi_clean'], pred, policy)
                tp, gt, npred = off['counts'][:3]
                counts[arm][c].append(bin_row(tp, gt, npred, off['matched_ann_ids'], heights))
                vec = r['raw'] if arm != 'FULL' else p_full
                cut = dict(F=CUT_F, RAW=model['cut_raw'], FULL=model['cut_full'])[arm]
                fast = bq.counts_at(m, vec, cut, groups)
                if fast[:3] != [tp, gt, npred]:
                    checks['fast_equals_official'] = False
                    checks['fast_mismatch'].append(dict(source=s, condition=c, arm=arm, fast=fast[:3], official=[tp, gt, npred]))
                elif fast[4] != counts[arm][c][-1][4]:
                    checks['tiny_from_alternative_matching'] += 1   # same counts, another maximum matching
        if (k + 1) % 200 == 0:
            print(f'{k + 1}/{len(sources)} sources ({time.perf_counter() - tic:.0f} s)', flush=True)
    C = {arm: {c: np.asarray(counts[arm][c], dtype=np.int64) for c in CONDS} for arm in ARMS}
    # references from the existing T4 dumps (same images; Clean = Kaggle T4, LR2 = Colab T4)
    refs, live_vs_dump = {}, {}
    if a.dump_lr2 and a.dump_clean:
        for c, d in (('lr2', a.dump_lr2), ('clean', a.dump_clean)):
            rr, b1m, ident = _ref_rows(d, c, sources, rows, policy, geoms)
            refs[c] = {k: np.asarray(v, dtype=np.int64) for k, v in rr.items()}
            same = sum(same_candidates(recs[(s, c)]['polygons'], recs[(s, c)]['raw'], b1m[s]['polygons'], b1m[s]['raw']) for s in sources)
            live_vs_dump[c] = dict(dump=ident, tasks=len(sources), identical_candidates=int(same),
                                   F_hmean_live=bq.hmean(C['F'][c].sum(0)), F_hmean_dump=bq.hmean(refs[c]['B1 .40/.60'].sum(0)))
    summ = {arm: {c: totals(C[arm][c]) for c in CONDS} for arm in ARMS}
    for arm in ARMS:
        summ[arm]['mean_hmean'] = (summ[arm]['clean']['hmean'] + summ[arm]['lr2']['hmean'])/2
    diffs = {}
    for x, y in (('FULL', 'F'), ('FULL', 'RAW'), ('RAW', 'F')):
        diffs[f'{x}_minus_{y}'] = dict(mean=bq.paired_boot_mean(C[x], C[y]),
                                       clean=p0.paired_boot(C[x]['clean'][:, :3], C[y]['clean'][:, :3]),
                                       lr2=p0.paired_boot(C[x]['lr2'][:, :3], C[y]['lr2'][:, :3]),
                                       tiny_pp={c: ((summ[x][c]['tiny_recall'] or 0) - (summ[y][c]['tiny_recall'] or 0))*100 for c in CONDS})
    # per-source description vs F (exact additive split of the Hmean difference, per condition)
    per_source = {}
    for c in CONDS:
        hF = bq.hmean(C['F'][c].sum(0))
        d_tp = (C['FULL'][c][:, 0] - C['F'][c][:, 0]).tolist()
        d_pr = (C['FULL'][c][:, 2] - C['F'][c][:, 2]).tolist()
        net = p0.net_contributions(d_tp, d_pr, hF)
        per_source[c] = dict(win=sum(v > 0 for v in net), tie=sum(v == 0 for v in net), loss=sum(v < 0 for v in net),
                             top10_share_of_net_gain=p0.top_share(net), delta_tp=int(sum(d_tp)), delta_pred=int(sum(d_pr)))
    gates, gate_detail = {}, {}
    if refs:
        ref_sum = {k: {c: totals(refs[c][k]) for c in CONDS} for k in REF_SINGLES}
        for k in REF_SINGLES:
            ref_sum[k]['mean_hmean'] = (ref_sum[k]['clean']['hmean'] + ref_sum[k]['lr2']['hmean'])/2
        best = max(REF_SINGLES, key=lambda k: ref_sum[k]['mean_hmean'])
        g1 = (summ['FULL']['mean_hmean'] - ref_sum[best]['mean_hmean'])*100
        g2 = (summ['FULL']['clean']['hmean'] - ref_sum['B-Clean .40/.60']['clean']['hmean'])*100
        g3 = ((summ['FULL']['lr2']['tiny_recall'] or 0) - (ref_sum['B1 .40/.60']['lr2']['tiny_recall'] or 0))*100
        gates = dict(g1_mean_ge_best_single_plus_0p5=bool(g1 >= GATE_MEAN_PP), g2_clean_ge_bclean_minus_0p3=bool(g2 >= GATE_CLEAN_PP),
                     g3_tiny_lr2_ge_b1_minus_3=bool(g3 >= GATE_TINY_PP))
        g1, g2, g3 = float(g1), float(g2), float(g3)
        gate_detail = dict(best_single=best, full_minus_best_single_mean_pp=g1,
                           full_minus_best_single_ci=bq.paired_boot_mean(C['FULL'], {c: refs[c][best] for c in CONDS}),
                           full_minus_bclean_clean_pp=g2, full_minus_b1_tiny_lr2_pp=g3, references=ref_sum)
    lat = read_json(a.latency) if a.latency else None
    lat_ratio = lat['paired']['ratio_of_medians'] if lat else None
    valid = (checks['consistency_all_true'] and checks['fast_equals_official'] and checks['nonfinite_features'] == 0 and
             checks['tasks'] == 2*len(sources) and bool(gates))
    m_ = diffs['FULL_minus_F']
    label, failed = outcome(valid, m_['mean']['delta_pp'], m_['mean']['ci95_pp'], m_['clean']['ci95_pp'][1], m_['lr2']['ci95_pp'][1],
                            m_['tiny_pp']['clean'], m_['tiny_pp']['lr2'], lat_ratio, gates or {'missing': False})
    fr = diffs['FULL_minus_RAW']['mean']
    res = dict(label='Box-quality scorer (frozen, TRAIN-fit) on development split A, Clean + LR2. Post-pilot user decision; '
                     'pilot label POSITIVE_BELOW_GATE unchanged. A was seen before: descriptive, cannot confirm an improvement.',
               spec=SPEC, subset=a.subset, sources=len(sources), split=SPLIT,
               frozen=dict(model_sha256=MODEL_SHA256, fit_sha256=FIT_SHA256, cut_full=model['cut_full'], cut_raw=model['cut_raw'],
                           cut_F=CUT_F, geometry=list(bq.GEOM), floor=bq.FLOOR),
               checks=checks, kept_boxes=kept_n, arms=summ, differences=diffs, per_source_vs_F=per_source,
               group_gates=gates, group_gate_detail=gate_detail, live_vs_t4_dump=live_vs_dump,
               latency=dict(file=str(a.latency) if a.latency else None, ratio_of_medians=lat_ratio,
                            ratio_of_p95=lat['paired']['ratio_of_p95'] if lat else None, memory=lat['memory'] if lat else None),
               outcome=label, small_useful_failed=failed,
               generalization_full_minus_raw=dict(reading=generalization(fr['delta_pp'], fr['ci95_pp']), pilot_delta_pp=.14,
                                                  pilot_ci95_pp=[.0004, .29], devA=fr))
    res = json.loads(json.dumps(res, default=_plain))
    save_json(out/'devA_result.json', res)
    show = dict(outcome=label, failed=failed, checks={k: v for k, v in checks.items() if k != 'fast_mismatch'},
                mean={arm: round(summ[arm]['mean_hmean']*100, 3) for arm in ARMS},
                FULL_minus_F=m_['mean'], FULL_minus_RAW=fr, gates=gates, generalization=res['generalization_full_minus_raw']['reading'])
    print(json.dumps(show, indent=1, default=str))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    e = sub.add_parser('extract')
    lt = sub.add_parser('latency')
    for s in (p, e, lt):
        for arg in ('--bundle', '--clean-root', '--prepared-root'):
            s.add_argument(arg, required=True, type=Path)
    for s in (p, e):
        s.add_argument('--subset', choices=('smoke', 'all'), default='smoke')
    for s in (e, lt):
        for arg in ('--model-config', '--weights-dir', '--output'):
            s.add_argument(arg, required=True, type=Path)
    e.add_argument('--shard', type=int, default=50)
    for arg in ('--model', '--fit'):
        lt.add_argument(arg, required=True, type=Path)
    lt.add_argument('--extract-dir', type=Path)
    lt.add_argument('--passes', type=int, default=2)
    v = sub.add_parser('evaluate')
    for arg in ('--bundle', '--extract-dir', '--model', '--fit', '--output'):
        v.add_argument(arg, required=True, type=Path)
    for arg in ('--dump-lr2', '--dump-clean', '--latency'):
        v.add_argument(arg, type=Path)
    v.add_argument('--subset', choices=('smoke', 'all'), default='all')
    a = ap.parse_args()
    if a.command in ('latency', 'extract'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    dict(prepare=prepare, extract=extract, latency=latency, evaluate=evaluate)[a.command](a)


if __name__ == '__main__':
    main()
