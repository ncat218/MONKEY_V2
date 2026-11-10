"""Render representative B1-cal image crops and masked P-maps on HPC."""
from rev4 import require
from rev4.b1_diagnostics import evidence_for
from rev4.calibration import CONDITIONS
from rev4.g4_review import crop_box


def save_sheet(path, selected, rows, tasks, infer_source, shrink):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from shapely.geometry import shape

    sheet = Image.new('RGB', (960, 180*len(selected)), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        source, ann_id = item['source_image_id'], item['ann_id']
        row, pair = rows[source], tasks[source]
        entry = next(gt for gt in row['gt_clean']['valid'] if str(gt['ann_id']) == ann_id)
        x0, y0, x1, y1 = shape(entry['geometry']).bounds
        pad = max(7., (y1-y0)*.55)
        bounds = x0-pad, y0-pad, x1+pad, y1+pad
        views = infer_source(source)
        require(ann_id in views['clean']['matched'] and ann_id not in views['lr2']['matched'],
                f'Review transition changed for {ann_id}')
        residual = dict(row, gt_clean=dict(row['gt_clean'], valid=[entry]))
        y = 180*index
        draw.text((8, y+5), f'{ann_id}  {item["primary"]}  LR2 height {item["native_height"]:.2f}px', fill='black')
        for col, condition in enumerate(CONDITIONS):
            task, view = pair[condition], views[condition]
            evidence, probability, valid = evidence_for(view, task, residual, shrink)
            coverage = evidence[ann_id]['coverage']
            require(coverage == item[f'coverage_{condition}'], f'Review P-map coverage changed for {ann_id}')
            native = 1/view['restore']
            rgb = cv2.imread(task['path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(rgb is not None, f'Unable to decode review image {task["sample_id"]}')
            photo = Image.fromarray(rgb[:, :, ::-1])
            photo = photo.crop(crop_box(bounds, native, photo.size))
            left, top, right, bottom = crop_box(bounds, native*view['factor'],
                                                 (probability.shape[1], probability.shape[0]))
            patch = np.where(valid[top:bottom, left:right], probability[top:bottom, left:right], 0.)
            pmap = Image.fromarray(np.rint(np.clip(patch, 0, 1)*255).astype(np.uint8)).convert('RGB')
            label = 'N/A' if coverage is None else f'{coverage:.2f}'
            for x, title, tile in ((480*col, f'{condition} image', photo),
                                   (480*col+240, f'{condition} P-map coverage {label}', pmap)):
                draw.text((x+6, y+25), title, fill='black')
                sheet.paste(ImageOps.contain(tile, (225, 128), method=Image.Resampling.NEAREST), (x+6, y+44))
    temporary = path.with_suffix('.png.tmp')
    sheet.save(temporary, format='PNG')
    temporary.replace(path)
