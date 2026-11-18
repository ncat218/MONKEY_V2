"""Automatic diagnostic proxies plus a stratified, explicitly pending human review."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely.geometry import shape, Polygon

from lowres.common import identity as fingerprint, read_json, save_json
from rev4 import require
from rev4.adaptation_plan import HEIGHT_NAMES
from rev4.decision_plan import REVIEW_SEED
from rev4.paired_eda import mode_result
from rev4.paired_eda_core import filtered_geometry, fp_transitions, overlap_index, overlaps, transition


def cached_record(root, model, task):
    identity = read_json(root/'models'/model['name']/'identity.json')
    require(identity['model'] == model['eda_model'], 'Historical EDA model identity mismatch')
    record = read_json(root/'models'/model['name']/task['condition']/f'{task["source_image_id"]}.json')
    require(record['identity_sha256'] == fingerprint(identity) and record['sample_id'] == task['sample_id'] and
            record['image_sha256'] == task['image_sha256'], 'Historical EDA image identity mismatch')
    return record


def score_bucket(score):
    return 'missing' if score is None else ('<0.50' if score < .5 else ('0.50-<0.70' if score < .7 else '>=0.70'))


class StratifiedReview:
    def __init__(self):
        self.populations = Counter()
        self.selected = {}

    def add(self, case):
        key = (case['condition'], case['group'], case['height_bin'], score_bucket(case['score']))
        self.populations[key] += 1
        rank = hashlib.sha256(f'{REVIEW_SEED}:{case["case_id"]}'.encode()).hexdigest()
        if key not in self.selected or rank < self.selected[key][0]:
            self.selected[key] = (rank, dict(case))

    def result(self):
        result = []
        for key in sorted(self.selected):
            case = self.selected[key][1]
            case.update(stratum=list(key), stratum_population=self.populations[key],
                        inclusion_probability=1/self.populations[key],
                        population_weight=self.populations[key], human_label='', reviewer='', notes='')
            result.append(case)
        return result


def fp_proxy(geometry, gt_index, matched, policy):
    if geometry is None:
        return dict(proxy='invalid_output', nearest_ann_id=None, max_iou=0., bounds=None)
    entries, shapes, _, tree = gt_index
    neighbors = []
    for j in tree.query(geometry, predicate='intersects'):
        gt = shapes[int(j)]
        inter = gt.intersection(geometry).area
        if inter:
            neighbors.append((inter/(gt.area+geometry.area-inter), entries[int(j)], inter/gt.area, inter/geometry.area))
    neighbors.sort(key=lambda item: (-item[0], item[1]))
    if not neighbors:
        label = 'no_valid_gt_overlap_background_or_missing_annotation'
    elif any(iou >= policy['match_iou'] and ann in matched for iou, ann, _, _ in neighbors):
        label = 'duplicate_or_assignment'
    elif sum(gt_fraction >= .5 for _, _, gt_fraction, _ in neighbors) >= 2:
        label = 'possible_merge'
    else:
        label = 'overlaps_gt_localization_or_partial'
    return dict(proxy=label, nearest_ann_id=neighbors[0][1] if neighbors else None,
                max_iou=neighbors[0][0] if neighbors else 0., bounds=list(geometry.bounds))


def prepare_review(a, models, rows, tasks, policy):
    from shapely.strtree import STRtree

    folder = a.output/'review'
    folder.mkdir(parents=True, exist_ok=True)
    model_map = {model['name']: model for model in models}
    review = StratifiedReview()
    proxy_counts, gt_counts, correspondence = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    fp_fields = ['case_id', 'source_image_id', 'condition', 'group', 'model', 'prediction_index', 'score',
                 'height_bin', 'proxy', 'nearest_ann_id', 'max_iou', 'bounds']
    gt_fields = ['case_id', 'source_image_id', 'condition', 'group', 'ann_id', 'height_bin', 'height_lr2',
                 'old_tp', 'new_tp', 'old_diagnosis', 'new_diagnosis', 'old_candidate_iou', 'new_candidate_iou',
                 'old_candidate_score', 'new_candidate_score', 'new_score_margin', 'new_kernel_coverage']
    with ((folder/'fp_taxonomy_proxies.csv').open('w', newline='', encoding='utf-8') as fp_file,
          (folder/'gt_diagnosis.csv').open('w', newline='', encoding='utf-8') as gt_file):
        fp_writer, gt_writer = csv.DictWriter(fp_file, fp_fields), csv.DictWriter(gt_file, gt_fields)
        fp_writer.writeheader(); gt_writer.writeheader()
        for number, task in enumerate(tasks, 1):
            source, condition = task['source_image_id'], task['condition']
            row = rows[source]
            old = cached_record(a.eda_root, model_map['B-Clean'], task)
            new = cached_record(a.eda_root, model_map['B1'], task)
            before, after = mode_result(old, 'common'), mode_result(new, 'common')
            require(old['gt_attributes'] == new['gt_attributes'], 'GT attributes changed across EDA models')
            entries = row['gt_clean']['valid']
            gt_shapes = [shape(entry['geometry']) for entry in entries]
            gt_index = ([str(entry['ann_id']) for entry in entries], gt_shapes, None, STRtree(gt_shapes))
            records = fp_transitions(before, after, row, policy)
            correspondence[condition].update(r['kind'] for r in records)
            for side, name, record, scored in [('old', 'B-Clean', old, before), ('new', 'B1', new, after)]:
                geometries, _ = filtered_geometry(scored['prediction'], row, policy)
                matched = {ann for ann, evidence in scored['instances'].items() if evidence['tp']}
                for fp in records:
                    index = fp[f'{side}_index']
                    if index is None:
                        continue
                    geometry = geometries.get(index)
                    proxy = fp_proxy(geometry, gt_index, matched, policy)
                    bounds = proxy['bounds']
                    height = (bounds[3]-bounds[1])*(max(1, row['clean_size'][1]//2)/row['clean_size'][1]) if bounds else 0.
                    bucket = HEIGHT_NAMES[sum(height >= cut for cut in (4, 8, 16))]
                    case = dict(case_id=f'{condition}:fp:{name}:{source}:{index}', source_image_id=source,
                            condition=condition, group=fp['kind'], model=name, prediction_index=index,
                            score=scored['prediction']['scores'][index], height_bin=bucket, **proxy)
                    fp_writer.writerow(dict(case, bounds=json.dumps(bounds)))
                    proxy_counts[condition, name].update([proxy['proxy']])
                    # Persistent FP: sample one paired case, rather than count its two views twice.
                    if fp['kind'] != 'persistent_fp' or side == 'new':
                        if bounds is not None:
                            review.add(dict(case, ann_id=None))
            for ann, old_evidence in before['instances'].items():
                current = after['instances'][ann]
                group = transition(old_evidence['tp'], current['tp'])
                attributes = new['gt_attributes'][ann]
                gt_counts[condition].update([group])
                gt_writer.writerow(dict(case_id=f'{condition}:gt:{source}:{ann}', source_image_id=source,
                    condition=condition, group=group, ann_id=ann, height_bin=attributes['height_bin'],
                    height_lr2=attributes['height_lr2'], old_tp=old_evidence['tp'], new_tp=current['tp'],
                    old_diagnosis=old_evidence['diagnosis'], new_diagnosis=current['diagnosis'],
                    old_candidate_iou=old_evidence['candidate_iou'], new_candidate_iou=current['candidate_iou'],
                    old_candidate_score=old_evidence['candidate_score'], new_candidate_score=current['candidate_score'],
                    new_score_margin=current['candidate_score_margin'], new_kernel_coverage=new['kernel_evidence'][ann]['coverage']))
                if group != 'persistent_tp':
                    review.add(dict(case_id=f'{condition}:gt:{source}:{ann}', source_image_id=source,
                        condition=condition, group=group, ann_id=ann, model='B1', prediction_index=None,
                        score=current['assigned']['score'] if current['assigned'] else current['candidate_score'],
                        height_bin=attributes['height_bin'], bounds=attributes['bounds'], proxy=current['diagnosis']))
            if number % 500 == 0:
                print(f'FP/GT diagnosis preparation: {number}/{len(tasks)} tasks', flush=True)
    cases = review.result()
    for index, case in enumerate(cases):
        case['panel_index'] = index
        case['image_pair'] = f'examples/{index:04d}.png'
    save_json(folder/'selected_review.json', dict(seed=REVIEW_SEED,
        rule='one minimum SHA256 rank per condition/group/height/score stratum', cases=cases,
        fp_height='FP polygon AABB projected to LR2; not GT-associated precision bins',
        manual_review_status='PENDING', invalid_fp_not_rendered=True))
    fields = ['case_id', 'condition', 'group', 'source_image_id', 'ann_id', 'model', 'prediction_index',
              'height_bin', 'score', 'stratum_population', 'inclusion_probability', 'population_weight',
              'image_pair', 'human_label', 'reviewer', 'notes']
    with (folder/'human_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(cases)
    summary = dict(fp_proxies=[dict(condition=c, model=m, counts=dict(v)) for (c, m), v in sorted(proxy_counts.items())],
                   gt_transitions={c: dict(v) for c, v in gt_counts.items()},
                   fp_correspondence={c: dict(v) for c, v in correspondence.items()},
                   sampled_cases=len(cases), manual_review_status='PENDING', proxies_are_not_causes=True)
    save_json(folder/'diagnosis_summary.json', summary)
    return cases, summary


def render_panels(a, model, task, row, probability, factor, prediction, cases):
    import cv2
    from PIL import Image, ImageDraw
    from rev4.stage_a import open_native_lr

    relevant = [case for case in cases if case['source_image_id'] == task['source_image_id'] and case['condition'] == task['condition']]
    if not relevant or model['name'] not in ('B-Clean', 'B1'):
        return
    native_factor = np.asarray(task['size'])/np.asarray(row['clean_size'])
    network = native_factor*factor
    pmap = probability[0].detach().cpu().numpy()
    panel_dir = a.output/'review/panels'
    panel_dir.mkdir(parents=True, exist_ok=True)
    with open_native_lr(Path(task['path']))[0] as image:
        rgb = image.convert('RGB')
        for case in relevant:
            x0, y0, x1, y1 = case['bounds']
            margin = max(8., .3*max(x1-x0, y1-y0))
            bounds = np.array([max(0, x0-margin), max(0, y0-margin),
                               min(row['clean_size'][0], x1+margin), min(row['clean_size'][1], y1+margin)])
            native_bounds = np.r_[np.floor(bounds[:2]*native_factor), np.ceil(bounds[2:]*native_factor)].astype(int)
            crop = rgb.crop(tuple(native_bounds))
            scale = min(480/max(crop.width, 1), 190/max(crop.height, 1))
            canvas = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))))
            draw = ImageDraw.Draw(canvas)

            def vertices(coords):
                return [tuple(p) for p in (np.asarray(coords).reshape(-1, 2)*native_factor-native_bounds[:2])*scale]

            window = Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3])])
            for entry in row['gt_clean']['valid']:
                geom = shape(entry['geometry'])
                if geom.intersects(window):
                    points = vertices(geom.exterior.coords)
                    draw.line(points+[points[0]], fill='lime', width=2)
            for polygon in prediction['polygons']:
                array = np.asarray(polygon).reshape(-1, 2)
                if len(array) >= 3 and np.isfinite(array).all() and Polygon(array).intersects(window):
                    points = vertices(array)
                    draw.line(points+[points[0]], fill='orange', width=2)
            lo = np.maximum(np.floor(bounds[:2]*network).astype(int), (0, 0))
            hi = np.minimum(np.ceil(bounds[2:]*network).astype(int), (pmap.shape[1], pmap.shape[0]))
            heat = pmap[lo[1]:hi[1], lo[0]:hi[0]]
            require(heat.size > 0, 'Empty review P-map crop')
            color = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1)*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            panel = Image.new('RGB', (480, 470), 'white')
            pd = ImageDraw.Draw(panel)
            pd.text((5, 4), f'{model["name"]} {case["condition"]} {case["group"]}', fill='black')
            pd.text((5, 22), 'Green: valid GT; Orange: prediction; P-map below', fill='black')
            pd.text((5, 40), 'Proxy only - human review PENDING', fill='black')
            panel.paste(canvas, (0, 70))
            panel.paste(Image.fromarray(color).resize(canvas.size), (0, 275))
            panel.save(panel_dir/f'{case["panel_index"]:04d}_{model["name"]}.png')


def combine_panels(a, cases):
    from PIL import Image

    root = a.output/'review'
    (root/'examples').mkdir(exist_ok=True)
    for case in cases:
        pair = Image.new('RGB', (960, 470), 'white')
        for index, name in enumerate(('B-Clean', 'B1')):
            path = root/'panels'/f'{case["panel_index"]:04d}_{name}.png'
            require(path.is_file(), f'Missing review panel: {path}')
            with Image.open(path) as panel:
                pair.paste(panel, (index*480, 0))
        pair.save(root/case['image_pair'])
