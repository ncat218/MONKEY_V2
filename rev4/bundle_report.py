"""One report for the research bundle: query-free zoom, seed replicates and PRC2 (development only).

Every comparison is against the fine-tuned baseline (B1 or B-Clean, seed 42) at the same fixed operating point.
Besides Hmean the report gives tiny-word recall (<4 px), <8 px recall and FP with paired source CIs, and the
tiny recall B1 itself reaches at the same FP by moving its score cutoff (matched-FP check).
"""
import numpy as np

from lowres.common import read_json, sha256
from rev4.bundle_plan import (EDSR_RECHECK_JOB, FUSION_RUN, MAIN_CONDITIONS, SETTINGS, STRESS_CONDITIONS,
                              STRESS_VARIANTS, TRAIN_ARMS, UNLABELED_SOURCES, ZOOM, ZOOM_MODELS, ZOOM_RULES,
                              setting_key)
from rev4.durable_io import save_json, write_text

PRIMARY = setting_key(*SETTINGS['primary'])
COMMON = setting_key(*SETTINGS['common'])
BIN_NAMES = ('<4', '4-<8', '8-<16', '>=16')
SEED = 20261107


def hmean(v):
    return 2*v[0]/(v[1]+v[2]) if v[1]+v[2] else 0.


def tiny(v):
    return v[4]/v[3] if v[3] else 0.


def lt8(v):
    return (v[4]+v[6])/(v[3]+v[5]) if v[3]+v[5] else 0.


def fp(v):
    return v[2]-v[0]


def ci(old, new, fn):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(2000):
        sel = rng.integers(0, len(old), len(old))
        values.append(fn(new[sel].sum(0))-fn(old[sel].sum(0)))
    return [float(v) for v in np.quantile(values, [.025, .975])]


def compare(old, new):
    """Paired source comparison of two (S, >=11) detail matrices: Hmean, tiny, <8 recall in pp; FP count."""
    o, n = old.sum(0), new.sum(0)
    out = {}
    for key, fn in (('hmean', hmean), ('tiny', tiny), ('lt8', lt8)):
        out[key] = dict(new=100*fn(n), delta=100*(fn(n)-fn(o)), ci=[100*x for x in ci(old, new, fn)])
    out['fp'] = dict(new=int(fp(n)), delta=int(fp(n)-fp(o)))
    return out


def detail_from_counts(task, key, fusion_pairs=None):
    """[tp, gt, pred, gt0, tp0, ..., gt3, tp3] from bundle eval records or fusion counts."""
    if fusion_pairs is None:
        return list(task['counts'][key])+list(task['bins'][key])
    index = fusion_pairs.index(tuple(float(x) for x in key.split('/')))
    return list(task['counts'][index])+[x for pair in task['bins'][index] for x in pair]


def load_fusion(ctx, name):
    from rev4.fusion_plan import PAIRS
    path = ctx.output.parent.parent/FUSION_RUN/'evaluation'/name/'counts.json'
    if not path.is_file():
        return None
    tasks = read_json(path)['tasks']
    return {t['sample_id']: t for t in tasks}, tuple(tuple(p) for p in PAIRS)


def frontier(ctx, condition):
    """B1 (seed 42) tiny and <8 recall as a function of FP over score cutoffs at mask .40 (fusion counts)."""
    loaded = load_fusion(ctx, 'B1')
    if loaded is None:
        return None
    tasks, pairs = loaded
    rows = []
    for pair in pairs:
        if pair[0] != .4:
            continue
        total = np.sum([detail_from_counts(t, setting_key(*pair), pairs) for t in tasks.values()
                        if t['condition'] == condition], axis=0)
        rows.append((fp(total), tiny(total), lt8(total), hmean(total)))
    rows.sort()
    return np.array(rows, dtype=np.float64)


def at_fp(front, value, column):
    """Interpolate B1's frontier at ``value`` FP; NaN outside the measured FP range (no extrapolation)."""
    if front is None or len(front) == 0 or not front[0, 0] <= value <= front[-1, 0]:
        return float('nan')
    return float(np.interp(value, front[:, 0], front[:, column]))


def matrix(tasks, condition, getter, exclude=()):
    return np.asarray([getter(t) for t in tasks if t['condition'] == condition and t['source_image_id'] not in exclude],
                      dtype=np.int64)


def zoom_section(ctx):
    sections = {}
    for name in ZOOM_MODELS:
        root = ctx.output/'zoom'/name
        if not (root/'records.json').exists():
            sections[name] = dict(status='MISSING')
            continue
        tasks = read_json(root/'records.json')['tasks']
        complete = (root/'complete.json').exists()
        conditions = [c for c in MAIN_CONDITIONS+STRESS_CONDITIONS if any(t['condition'] == c for t in tasks)]
        variants = {}
        for variant in dict.fromkeys(v for t in tasks for v in t['results']):
            variants[variant] = {}
            for condition in conditions:
                subset = [t for t in tasks if t['condition'] == condition and variant in t['results']]
                if not subset:
                    continue
                base = matrix(subset, condition, lambda t: t['results']['global'][PRIMARY])
                cur = matrix(subset, condition, lambda t: t['results'][variant][PRIMARY])
                row = compare(base, cur)
                row['fp_lt8'] = int(cur.sum(0)[11]+cur.sum(0)[12]) if cur.shape[1] >= 15 else None
                row['common_hmean'] = 100*hmean(matrix(subset, condition, lambda t: t['results'][variant][COMMON]).sum(0))
                if condition in MAIN_CONDITIONS:
                    keep = [t for t in subset if t['source_image_id'] not in UNLABELED_SOURCES]
                    b2 = matrix(keep, condition, lambda t: t['results']['global'][PRIMARY])
                    c2 = matrix(keep, condition, lambda t: t['results'][variant][PRIMARY])
                    row['delta_hmean_excluding_flagged'] = 100*(hmean(c2.sum(0))-hmean(b2.sum(0)))
                ms = []
                for t in subset:
                    if variant in ZOOM_RULES:
                        source, _, restoration = ZOOM_RULES[variant]
                        ms.append(t['ms']['global_pass']+t['ms'][f'{source}_{restoration}']+
                                  (t['ms'].get('guided_select', 0.) if source == 'guided' else 0.))
                    else:
                        ms.append(t['ms'].get(variant, t['ms']['global_pass']))
                row['median_ms'] = float(np.median(ms))
                row['sources'] = len(subset)
                variants[variant][condition] = row
        sections[name] = dict(status='COMPLETE' if complete else 'PARTIAL', tasks=len(tasks), variants=variants)
    return sections


def training_section(ctx):
    out = {}
    refs = {'B1': load_fusion(ctx, 'B1'), 'B-Clean': load_fusion(ctx, 'B-Clean')}
    for arm in TRAIN_ARMS:
        path = ctx.output/'train'/arm/'evaluation_summary.json'
        if not path.exists():
            out[arm] = dict(status='MISSING')
            continue
        summary = read_json(path)
        reference = 'B-Clean' if arm == 'BClean_s43' else 'B1'
        selected = read_json(summary['selected_record'])['tasks']
        stress = read_json(summary['stress_record'])['tasks']
        comparisons = {}
        ref = refs[reference]
        if ref is not None:
            ref_tasks, pairs = ref
            for condition in MAIN_CONDITIONS+STRESS_CONDITIONS:
                source = selected if condition in MAIN_CONDITIONS else stress
                subset = [t for t in source if t['condition'] == condition]
                if not subset:
                    continue
                new = np.asarray([detail_from_counts(t, PRIMARY) for t in subset], dtype=np.int64)
                old = np.asarray([detail_from_counts(ref_tasks[t['sample_id']], PRIMARY, pairs) for t in subset], dtype=np.int64)
                comparisons[condition] = compare(old, new)
        out[arm] = dict(status='COMPLETE', summary=summary, reference=f'{reference} seed 42', comparisons=comparisons)
    return out


def fmt(value):
    return 'N/A' if value is None or value != value else f'{value:.2f}'


def dci(item):
    lo, hi = item['ci']
    return f'{item["delta"]:+.2f} [{lo:+.2f}; {hi:+.2f}]'


def report(ctx):
    status = read_json(ctx.output/'status.json') if (ctx.output/'status.json').exists() else {}
    preflight = read_json(ctx.output/'preflight.json') if (ctx.output/'preflight.json').exists() else {}
    zoom = zoom_section(ctx)
    training = training_section(ctx)
    fronts = {c: frontier(ctx, c) for c in MAIN_CONDITIONS}
    lines = ['# Research bundle v1 · query-free zoom, seed replicates, PRC2', '',
             'One Slurm allocation. 2,262 remaining-validation sources (development, already used), Clean and native LR2;',
             'stress conditions on the hashed 500-source cohort of the fusion study.',
             f'Baseline = fine-tuned B1 (or B-Clean), seed 42, at the same fixed point {PRIMARY} (secondary {COMMON}). No TEST, no new threshold search.',
             'Δ in percentage points with paired source bootstrap 95% CI (2,000 draws, seed 20261107). Tiny = GT height <4 px referenced to LR2.',
             f'Whole-image EDSR / bicubic on fine-tuned B1 is job {EDSR_RECHECK_JOB} (.runtime/edsr_recheck/v1/results); this bundle only adds EDSR on zoom crops.', '',
             '## Phase status', '', '| Phase | Status |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in status.items()]
    lines += ['', '## 1. Query-free zoom (locate weak or small text → crop → magnify ×2 → detect again)', '',
              f'Zoom owns words with LR2-referenced height < {ZOOM["tau_ref"]} px inside a window; at most {ZOOM["k_max"]} half-size windows. '
              'guided = windows from the detector\'s own weak/small P-map evidence; oracle = windows on small GT words (upper bound, '
              'not deployable); tile2x2 = uniform quadrants; guided_edsr = guided windows restored by EDSR x2 before detection. '
              'replace / union / verify = fusion rule (tile2x2_union reproduces Trí\'s ZB1). '
              '"B1 tiny @ same FP" = tiny recall B1 reaches by moving its score cutoff to the same FP (mask .40, fusion counts).', '']
    for name, section in zoom.items():
        if section.get('status') not in ('COMPLETE', 'PARTIAL'):
            lines += [f'**{name}: zoom records missing.**', '']
            continue
        for condition in MAIN_CONDITIONS:
            lines += [f'### {name} · {condition} at {PRIMARY} ({section["status"]})', '',
                      '| Variant | H | ΔH | Tiny | ΔTiny | B1 tiny @ same FP | <8 px | Δ<8 px | FP (Δ) | FP of boxes <8 px | ΔH excl. flagged source | H at .30/.40 | Median ms |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for variant, rows in section['variants'].items():
                if condition not in rows:
                    continue
                r = rows[condition]
                same = at_fp(fronts.get(condition), r['fp']['new'], 1)*100 if name == 'B1' else float('nan')
                lines.append(f'| {variant} | {r["hmean"]["new"]:.2f} | {dci(r["hmean"])} | {r["tiny"]["new"]:.2f} | {dci(r["tiny"])} | '
                             f'{fmt(same)} | {r["lt8"]["new"]:.2f} | {dci(r["lt8"])} | {r["fp"]["new"]} ({r["fp"]["delta"]:+d}) | '
                             f'{r["fp_lt8"] if r["fp_lt8"] is not None else "N/A"} | {fmt(r.get("delta_hmean_excluding_flagged"))} | '
                             f'{r["common_hmean"]:.2f} | {r["median_ms"]:.1f} |')
            lines.append('')
    b1 = zoom.get('B1', {})
    if b1.get('status') in ('COMPLETE', 'PARTIAL'):
        lines += ['### B1 zoom under stress at .40/.60 (500 sources, Hmean; Δ vs B1 global with CI)', '',
                  '| Variant | ' + ' | '.join(STRESS_CONDITIONS) + ' |', '|---|' + '---:|'*len(STRESS_CONDITIONS)]
        for variant in STRESS_VARIANTS:
            v = b1['variants'].get(variant, {})
            cells = []
            for condition in STRESS_CONDITIONS:
                if condition not in v:
                    cells.append('N/A')
                    continue
                h = v[condition]['hmean']
                cells.append(f'{h["new"]:.2f}' if variant == 'global' else f'{h["new"]:.2f} ({dci(h)})')
            lines.append(f'| {variant} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## 2. Seed replicates and PRC2 (B0 init, 10k updates, same recipe as B1/B-Clean)', '',
              '| Arm | Seed | Selected | vs | ΔClean H | ΔLR2 H | ΔLR2 tiny | ΔLR2 <8 px | LR2 FP (Δ) | Calibrated (mask/score) | Cal Clean / LR2 H |',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|']
    for arm, item in training.items():
        if item['status'] != 'COMPLETE':
            lines.append(f'| {arm} | | {status.get("train:"+arm, "MISSING")} | | | | | | | | |')
            continue
        s, comp = item['summary'], item['comparisons']
        cal = s['calibrated']
        clean, lr2 = comp.get('clean'), comp.get('lr2')
        lr2_fp = f'{lr2["fp"]["new"]} ({lr2["fp"]["delta"]:+d})' if lr2 else 'N/A'
        lines.append(f'| {arm} | {s["training"]["seed"]} | {s["native_selection"]["selected_updates"]:,} | {item["reference"]} | '
                     f'{dci(clean["hmean"]) if clean else "N/A"} | {dci(lr2["hmean"]) if lr2 else "N/A"} | '
                     f'{dci(lr2["tiny"]) if lr2 else "N/A"} | {dci(lr2["lt8"]) if lr2 else "N/A"} | '
                     f'{lr2_fp} | {cal["mask_thr"]:.2f}/{cal["min_text_score"]:.2f} | '
                     f'{100*cal["scores"]["clean"]["hmean"]:.2f} / {100*cal["scores"]["lr2"]["hmean"]:.2f} |')
    lines += ['', '### Stress transfer of the new checkpoints (ΔHmean vs the seed-42 reference, 500 sources)', '',
              '| Arm | ' + ' | '.join(STRESS_CONDITIONS) + ' |', '|---|' + '---:|'*len(STRESS_CONDITIONS)]
    for arm, item in training.items():
        if item['status'] == 'COMPLETE':
            lines.append(f'| {arm} | ' + ' | '.join(dci(item['comparisons'][c]['hmean']) if c in item['comparisons'] else 'N/A'
                                                     for c in STRESS_CONDITIONS) + ' |')
    prc2 = preflight.get('prc2', {})
    lines += ['', '## 3. PRC v1 replay and PRC2 gate', '']
    if 'summary' in prc2:
        p = prc2['summary']
        lines += [f'Probe {prc2["status"]} on {p["samples"]} LR2 TRAIN samples. Alignment correlation median {fmt(p["alignment_corr_median"])} '
                  f'(flipped control {fmt(p["flipped_corr_median"])}). Teacher P-map median: text {fmt(p["teacher_text_mean_median"])}, '
                  f'background {fmt(p["teacher_bg_mean_median"])}.',
                  f'v1 path: teacher predictions/crop {p["v1_predictions_mean"]:.2f} (largest box covers a median {fmt(p["v1_pred_area_frac_max_median"])} '
                  f'of the crop), matched words {p["v1_matched_mean"]:.2f}, GT with any prediction IoU≥.5 {p["v1_best_iou_ge_05_mean"]:.2f}, '
                  f'dropped by ROI {p["v1_ignored_roi_mean"]:.2f}, by ignore {p["v1_ignored_annotations_mean"]:.2f}; valid GT/crop {p["gt_valid_mean"]:.2f}.',
                  f'v2 kernel coverage: selected {p["v2_selected_mean"]:.2f} of {p["v2_eligible_mean"]:.2f} eligible kernels; '
                  f'active in {p["v2_active_fraction"]:.0%} of samples. Criteria: {prc2.get("criteria")}.']
    elif prc2:
        lines.append('PRC2 probe raised an error; see preflight.json.')
    if training.get('PRC2', {}).get('status') == 'COMPLETE':
        audit = training['PRC2']['summary']['training']['audit']
        lines.append(f'PRC2 training: active in {audit["active"]:,}/{audit["microbatches"]:,} microbatches '
                     f'({audit["active_fraction"]:.1%}); mean selected kernels {audit["selected_kernels_mean"]:.2f}.')
    zoom_probe = preflight.get('zoom', {})
    if zoom_probe:
        lines += ['', f'Zoom probe: {zoom_probe.get("status")}; sentinels {zoom_probe.get("sentinels")}.']
    lines += ['', '## Limits', '',
              '- Development cohort reused many times; CIs are paired source bootstraps, not seed variance (seed 43 rows give one replicate).',
              '- Zoom keeps the frozen detector, postprocessor and thresholds; oracle windows use GT and are an upper bound only.',
              '- Latency is a non-isolated per-image screen inside the job (crop restoration + model passes), not a deployment benchmark.',
              f'- The flagged-source column removes {", ".join(UNLABELED_SOURCES)} (newspaper body text without labels, found by Trí before this run).',
              '- PRC2 changes how PRC selects kernels (post-hoc amendment after PRC v1 was found inactive); report it as such.', '']
    root = ctx.output/'results'
    root.mkdir(parents=True, exist_ok=True)
    write_text(root/'bundle_report.md', '\n'.join(lines))
    save_json(root/'bundle_summary.json', dict(identity=ctx.trial_identity, status=status, zoom=zoom, training=training,
                                               frontier={c: (f.tolist() if f is not None else None) for c, f in fronts.items()},
                                               prc2_probe=prc2.get('summary'), no_test=True))
    save_json(root/'complete.json', dict(identity=ctx.trial_identity, report_sha256=sha256(root/'bundle_report.md'),
                                         final=all(v == 'PASS' or str(v).startswith('SKIPPED') for v in status.values()),
                                         no_test=True))
    print('Bundle report:', root/'bundle_report.md', flush=True)
