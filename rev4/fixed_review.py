"""Fixed-setting review cards with readable Clean context and explicit target."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, shape

from rev4 import require
from rev4.stage_a import open_native_lr

WIDTH, HEIGHT = 640,760


def valid_panel(path):
    try:
        with Image.open(path) as image:
            require(image.size==(WIDTH,HEIGHT),'Unexpected fixed panel size')
            image.verify()
        return True
    except (OSError,ValueError):
        return False


def render(a,model,task,clean_task,row,probability,factor,prediction,cases):
    selected=[c for c in cases if (c['source_image_id'],c['condition'])==(task['source_image_id'],task['condition'])]
    folder=a.output/'review/panels'
    folder.mkdir(parents=True,exist_ok=True)
    native_factor=np.asarray(task['size'])/np.asarray(row['clean_size'])
    network=native_factor*factor
    pmap=probability[0].detach().cpu().numpy()
    with open_native_lr(Path(task['path']))[0] as source,open_native_lr(Path(clean_task['path']))[0] as clean_source:
        native,clean=source.convert('RGB'),clean_source.convert('RGB')
        for case in selected:
            x0,y0,x1,y1=case['bounds']
            margin=max(12.,.5*max(x1-x0,y1-y0))
            bounds=np.array([max(0,x0-margin),max(0,y0-margin),
                             min(row['clean_size'][0],x1+margin),min(row['clean_size'][1],y1+margin)])
            target=None
            if case.get('ann_id') is not None:
                entry=next(e for e in row['gt_clean']['valid'] if str(e['ann_id'])==case['ann_id'])
                target=shape(entry['geometry'])
            else:
                # FP target belongs to case.model; use the same clean-coordinate rectangle
                # in both panels even when the other model has no corresponding prediction.
                target=Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
            window=Polygon([(bounds[0],bounds[1]),(bounds[2],bounds[1]),
                            (bounds[2],bounds[3]),(bounds[0],bounds[3])])
            def rgb_crop(image,ratio,show_predictions):
                lo=np.floor(bounds[:2]*ratio).astype(int)
                hi=np.ceil(bounds[2:]*ratio).astype(int)
                crop=image.crop(tuple(np.r_[lo,hi]))
                resize=min(WIDTH/max(crop.width,1),190/max(crop.height,1))
                canvas=crop.resize((max(1,round(crop.width*resize)),max(1,round(crop.height*resize))))
                actual_scale=np.asarray(canvas.size)/np.asarray(crop.size)
                draw=ImageDraw.Draw(canvas)
                def line(geometry,color,width=1):
                    if geometry.is_empty:
                        return
                    if geometry.geom_type!='Polygon':
                        for part in getattr(geometry,'geoms',[]):
                            line(part,color,width)
                        return
                    points=[tuple(p) for p in (np.asarray(geometry.exterior.coords)*ratio-lo)*actual_scale]
                    draw.line(points,fill=color,width=width)
                for entry in row['gt_clean']['valid']:
                    geom=shape(entry['geometry'])
                    if geom.intersects(window): line(geom,'lime',2)
                for entry in row['gt_clean']['ignored']:
                    geom=shape(entry['geometry'])
                    if geom.intersects(window): line(geom,'cyan',2)
                if show_predictions:
                    for polygon in prediction['polygons']:
                        array=np.asarray(polygon).reshape(-1,2)
                        if len(array)<3 or not np.isfinite(array).all(): continue
                        geom=Polygon(array)
                        if not geom.is_valid: geom=geom.buffer(0)
                        if not geom.is_empty and geom.intersects(window): line(geom,'orange',2)
                line(target,'magenta',2)
                return canvas
            clean_crop=rgb_crop(clean,np.asarray([1.,1.]),False)
            lr_crop=rgb_crop(native,native_factor,True)
            lo=np.maximum(np.floor(bounds[:2]*network).astype(int),(0,0))
            hi=np.minimum(np.ceil(bounds[2:]*network).astype(int),(pmap.shape[1],pmap.shape[0]))
            heat=pmap[lo[1]:hi[1],lo[0]:hi[0]]
            require(heat.size>0,'Empty fixed review P-map crop')
            color=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
            heat_image=Image.fromarray(color).resize(lr_crop.size)
            panel=Image.new('RGB',(WIDTH,HEIGHT),'white')
            draw=ImageDraw.Draw(panel)
            draw.text((4,2),f'{model["name"]} {task["condition"]} mask .40 / score .60; {case["group"]}',fill='black')
            draw.text((4,19),f'Target: {case["case_id"]}',fill='black')
            draw.text((4,36),'GT valid=green, ignored=cyan, prediction=orange, target=magenta',fill='black')
            draw.text((4,57),'Clean reference (same source/FOV)',fill='black')
            panel.paste(clean_crop,(0,75))
            draw.text((4,280),f'Actual {task["condition"]} input with fixed-setting predictions',fill='black')
            panel.paste(lr_crop,(0,300))
            draw.text((4,507),'P-map; diagnostic kernel cutoff = .40; visual review PENDING',fill='black')
            panel.paste(heat_image,(0,530))
            path=folder/f'{case["panel_index"]:04d}_{model["name"]}.png'
            temp=path.with_suffix('.tmp.png')
            panel.save(temp)
            temp.replace(path)


def combine(a,cases):
    root=a.output/'review'
    (root/'examples').mkdir(parents=True,exist_ok=True)
    for case in cases:
        pair=Image.new('RGB',(2*WIDTH,HEIGHT),'white')
        for index,name in enumerate(('B-Clean','B1')):
            path=root/'panels'/f'{case["panel_index"]:04d}_{name}.png'
            require(valid_panel(path),f'Incomplete fixed review panel: {path}')
            with Image.open(path) as panel:
                pair.paste(panel,(index*WIDTH,0))
        pair.save(root/case['image_pair'])
