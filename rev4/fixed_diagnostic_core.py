"""Fixed .40/.60 diagnostics; frozen evaluator and historical helpers stay intact."""
from collections import Counter

import cv2
import numpy as np
from shapely.geometry import Polygon, shape

from lowres.inference import paint
from rev4 import require
from rev4.paired_eda_core import assert_transition_counts, transition

MASK = .40
SCORE = .60
PAIR_KEY = '0.40/0.60'


def kernel_at_threshold(probability, img_shape, scale_factor, task, gt, native_size,
                        shrink_ratio, offset, mask_threshold=MASK):
    """Same kernel geometry/rasterization as legacy evidence; explicit P-map cutoff."""
    require(0 < mask_threshold < 1, 'Invalid diagnostic P-map threshold')
    clean = np.asarray(gt['clean_size'])
    network = np.asarray(task['size'])/clean*scale_factor
    native_y = native_size[1]/clean[1]
    height, width = img_shape
    valid = np.zeros(probability.shape,dtype=np.uint8)
    paint(valid,shape(gt['roi_clean']),network,1)
    valid[height:,:] = 0
    valid[:,width:] = 0
    for entry in gt['gt_clean']['ignored']:
        paint(valid,shape(entry['geometry']),network,0)
    above = probability > mask_threshold
    evidence = {}
    for entry in gt['gt_clean']['valid']:
        geometry = shape(entry['geometry'])
        y0,y1 = geometry.bounds[1::2]
        item = dict(native_height=(y1-y0)*native_y,network_input_height=(y1-y0)*network[1],
                    kernel_pixels=0,coverage=None)
        evidence[str(entry['ann_id'])] = item
        if geometry.geom_type != 'Polygon' or geometry.interiors:
            continue
        points = (np.asarray(geometry.exterior.coords[:-1])*network).astype(np.int32)
        polygon = Polygon(points)
        distance = int(polygon.area*(1-shrink_ratio**2)/(polygon.length+.001)+.5)
        shrunk = np.asarray(offset(points,-distance))
        if shrunk.size < 6 or shrunk.size % 2:
            continue
        kernel = shrunk.reshape(-1,2).astype(np.int32)
        x0,y0 = np.maximum(kernel.min(axis=0),(0,0))
        x1,y1 = np.minimum(kernel.max(axis=0)+1,(width,height))
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1-y0,x1-x0),dtype=np.uint8)
        cv2.fillPoly(local,[kernel-np.array([x0,y0],dtype=np.int32)],1)
        selected = local.astype(bool) & valid[y0:y1,x0:x1].astype(bool)
        size = int(selected.sum())
        item['kernel_pixels'] = size
        if size >= 4:
            item['coverage'] = float(np.count_nonzero(above[y0:y1,x0:x1] & selected)/size)
    return evidence,valid,above


def assert_reference(result, attributes, expected):
    ids = [ann for ann,info in attributes.items() if info['height_lr2'] < 4]
    observed = list(result['counts'][:3])+[len(ids),sum(result['instances'][ann]['tp'] for ann in ids)]
    require(observed == list(map(int,expected)),
            f'Fixed .40/.60 differs from original sweep: got {observed}, expected {list(expected)}')


def instance_transitions(before, after):
    require(before['instances'].keys() == after['instances'].keys(),'Transition GT IDs differ')
    counts = Counter(transition(item['tp'],after['instances'][ann]['tp'])
                     for ann,item in before['instances'].items())
    assert_transition_counts(counts,before['counts'][0],after['counts'][0],before['counts'][1])
    return dict(counts)
