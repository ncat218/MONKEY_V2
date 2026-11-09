"""Create a local-to-HPC review sheet from the existing LR2 pilot artifacts."""
import math
from pathlib import Path

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
    save_json(output/'evidence.json', dict(source_job=a.s2a_job_id, examples=evidence))
    print(output/'lr2_pmap_review.png')


if __name__ == '__main__':
    main()
