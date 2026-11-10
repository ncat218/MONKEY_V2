"""Create a local-to-HPC review sheet from the existing LR2 pilot artifacts."""
import math
from pathlib import Path
from statistics import median

from lowres.common import jsonl, read_json, save_json
from rev4 import require
from rev4.stage_a import rows_for
from submit_s2b import verified_inputs


def representative_cases(losses):
    cases = []
    for category, lo, hi, midpoint in (
        ('weak_response', 6, 8, 7), ('weak_response', 8, 16, 12),
        ('weak_response', 16, 32, 24), ('geometry', 8, 16, 12),
    ):
        choices = [x for x in losses if x['primary'] == category and lo <= x['native_height'] < hi]
        if choices:
            cases.append(min(choices, key=lambda x: (abs(x['native_height']-midpoint),
                                                      x['source_image_id'], x['ann_id'])))
    return cases


def crop_box(bounds, factors, size):
    x0, y0, x1, y1 = bounds
    sx, sy = factors
    return (max(0, math.floor(x0*sx)), max(0, math.floor(y0*sy)),
            min(size[0], math.ceil(x1*sx)), min(size[1], math.ceil(y1*sy)))


def diagnostic_summary(summary):
    lr2 = summary['conditions']['lr2']
    losses = lr2['failures']
    strata = []
    for height in (0, 4, 6, 8):
        cohort = [x for x in losses if x['native_height'] >= height and x['kernel_pixels'] >= 4]
        weak = [x for x in cohort if x['primary'] == 'weak_response']
        clean_coverages = [x['coverage_clean'] for x in cohort if x['coverage_clean'] is not None]
        strata.append(dict(min_native_height=height, valid_kernel_misses=len(cohort), weak_response=len(weak),
                           weak_response_sensitivity=sum(x['sensitivity'] == 'weak_response' for x in cohort),
                           clean_coverage_ge_030=sum(x >= .30 for x in clean_coverages),
                           clean_coverage_ge_050=sum(x >= .50 for x in clean_coverages),
                           weak_median_network_height=median(x['network_input_height'] for x in weak) if weak else None))
    return dict(review_status='PENDING_HUMAN_REVIEW', clean_tp_to_fn=len(losses),
                invalid_lr2_kernels=sum(x['kernel_pixels'] < 4 for x in losses), strata=strata,
                primary_failure_counts=lr2['primary_failure_counts'], native_height_bins=lr2['native_height_bins'],
                b_up_gain_pp=summary['b_up_lr2_gain_pp'], b_up_gain_95ci_pp=summary['b_up_lr2_gain_95ci_pp'],
                b_up_recovery_ratio=summary['b_up_lr2_recovery_ratio'])


def markdown(evidence):
    lo, hi = evidence['b_up_gain_95ci_pp']
    recovery = evidence['b_up_recovery_ratio']
    lines = ['# Rev.4 G4 review for LR2', '',
             f'Human review pending. S2-A evidence from job {evidence["source_job"]}.', '',
             f'B_up Hmean gain {evidence["b_up_gain_pp"]:.2f} pp (95% CI {lo:.2f} to {hi:.2f} pp).',
             ('B_up recovery ratio N/A.' if recovery is None else f'B_up recovered {recovery:.1%} of the Hmean drop.'),
             f'Clean TP to LR2 FN {evidence["clean_tp_to_fn"]:,}, including '
             f'{evidence["invalid_lr2_kernels"]:,} invalid LR2 kernels.', '',
             '| Minimum native height | Misses with valid kernel | Weak at c0=0.30 | Weak at c0=0.50 | '
             'Clean coverage ≥0.30 | Clean coverage ≥0.50 |', '|---:|---:|---:|---:|---:|---:|']
    for item in evidence['strata']:
        lines.append(f'| {item["min_native_height"]} px | {item["valid_kernel_misses"]:,} | '
                     f'{item["weak_response"]:,} | {item["weak_response_sensitivity"]:,} | '
                     f'{item["clean_coverage_ge_030"]:,} | {item["clean_coverage_ge_050"]:,} |')
    lines += ['', 'See `lr2_pmap_review.png` and `evidence.json` for representative cases. '
             'Use G4 evidence to review response, scale and teacher suitability before choosing M.', '']
    return '\n'.join(lines)


def main():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from shapely.geometry import shape

    a = verified_inputs()
    run = a.s2a_run
    selected = representative_cases(a.s2a_summary['conditions']['lr2']['failures'])
    require(selected, 'No representative LR2 misses in S2-A')
    ids = {row['source_image_id'] for row in selected}
    rows = rows_for(a, ids)
    tasks = {(item['condition'], item['source_image_id']): item for item in jsonl(run/'prepared/tasks.jsonl')
             if item['source_image_id'] in ids}
    sheet = Image.new('RGB', (1050, 172*len(selected)), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    evidence = []
    for i, row in enumerate(selected):
        source = row['source_image_id']
        ann_id = row['ann_id']
        geometry = next(ann['geometry'] for ann in rows[source]['gt_clean']['valid'] if str(ann['ann_id']) == ann_id)
        x0, y0, x1, y1 = shape(geometry).bounds
        pad = max(7., (y1-y0)*.55)
        bounds = x0-pad, y0-pad, x1+pad, y1+pad
        pictures, observations = [], {}
        for condition in ('clean', 'lr2', 'b_up_lr2'):
            parent = run/'results/samples'/condition
            sample = read_json(parent/f'{source}.json')
            resize = sample['resize']
            native = tuple(1/x for x in resize['lr_to_clean'])
            network = tuple(x*y for x, y in zip(native, resize['scale_factor']))
            if condition in ('clean', 'lr2'):
                image = cv2.imread(tasks[condition, source]['path'],
                                   cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                require(image is not None, f'Image could not be decoded {source} {condition}')
                rgb = Image.fromarray(image[:, :, ::-1])
                pictures.append((f'{condition} image', rgb.crop(crop_box(bounds, native, rgb.size))))
            with np.load(parent/f'{source}.npz', allow_pickle=False) as data:
                p = data['p'].astype(np.float32)
                valid = np.unpackbits(data['valid'], count=p.size).reshape(p.shape).astype(bool)
            left, top, right, bottom = crop_box(bounds, network, (p.shape[1], p.shape[0]))
            region = np.where(valid[top:bottom, left:right], p[top:bottom, left:right], 0)
            pmap = Image.fromarray(np.rint(np.clip(region, 0, 1)*255).astype(np.uint8)).convert('RGB')
            coverage = sample['gt'][ann_id]['coverage']
            label = 'N/A' if coverage is None else f'{coverage:.2f}'
            pictures.append((f'{condition} P map C={label}', pmap))
            observations[condition] = dict(coverage=coverage,
                                           matched=ann_id in set(map(str, sample['metrics']['matched_ann_ids'])))
        evidence.append(dict(source_id=source, ann_id=ann_id, native_height=row['native_height'],
                             category=row['primary'], conditions=observations))
        y = i*172
        draw.text((8, y+5), f'{source} {row["primary"]} native h={row["native_height"]:.2f}px', fill='black')
        for col, (label, picture) in enumerate(pictures):
            x = col*210
            draw.text((x+5, y+25), label, fill='black')
            sheet.paste(ImageOps.contain(picture, (196, 112), method=Image.Resampling.NEAREST), (x+5, y+45))
    output = Path(a.s2a_run).parents[1]/'rev4_b1/g4_review'
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output/'lr2_pmap_review.png', format='PNG')
    report = dict(source_job=a.s2a_job_id, s2a_summary_sha256=a.s2a_summary_sha256, examples=evidence,
                  **diagnostic_summary(a.s2a_summary))
    save_json(output/'evidence.json', report)
    (output/'g4_report.md').write_text(markdown(report), encoding='utf-8')
    print(output/'g4_report.md', output/'lr2_pmap_review.png', sep='\n')


if __name__ == '__main__':
    main()
