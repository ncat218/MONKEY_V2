"""Pinned CRAFT preprocessing/quad decoding and common evaluator coordinates."""
import numpy as np
import cv2

from rev4 import require
from rev4.craft_vendor import craft_utils,imgproc_reference

SETTINGS=dict(canvas_size=1280,mag_ratio=1.5,text_threshold=.7,low_text=.4,link_threshold=.4,
              poly=False,link_refiner=False,precision='fp32',threshold_selection='official defaults; no tuning')


def preprocess(rgb):
    require(rgb.ndim==3 and rgb.shape[2]==3 and rgb.dtype==np.uint8,'Expected RGB uint8 input')
    resized,ratio,_=imgproc_reference.resize_aspect_ratio(rgb,SETTINGS['canvas_size'],
        interpolation=cv2.INTER_LINEAR,mag_ratio=SETTINGS['mag_ratio'])
    h,w=rgb.shape[:2]
    rh,rw=int(h*ratio),int(w*ratio)
    require(rh>0 and rw>0,'Invalid CRAFT resized dimensions')
    normalized=imgproc_reference.normalizeMeanVariance(resized)
    return np.ascontiguousarray(normalized.transpose(2,0,1)),dict(native_hw=[h,w],
        resized_hw=[rh,rw],padded_hw=list(resized.shape[:2]),nominal_ratio=float(ratio),
        actual_scale_xy=[rw/w,rh/h],map_stride=2)


def map_boxes_to_clean(boxes,metadata,clean_size):
    rh,rw=metadata['resized_hw']
    factor=np.asarray(clean_size,dtype=np.float64)/np.asarray([rw,rh])*2
    return [(np.asarray(box,dtype=np.float64)*factor).reshape(-1).tolist() for box in boxes]


def decode(region,affinity,metadata,clean_size):
    require(region.shape==affinity.shape and region.ndim==2 and np.isfinite(region).all() and
            np.isfinite(affinity).all(),'CRAFT score maps must be finite and aligned')
    boxes,labels,components=craft_utils.getDetBoxes_core(region,affinity,
        SETTINGS['text_threshold'],SETTINGS['link_threshold'],SETTINGS['low_text'])
    # Regression heatmaps can exceed 1; matching does not depend on confidence.
    # Clipped component maximum satisfies the unchanged evaluator score schema.
    scores=[float(np.clip(region[labels==component].max(),0,1)) for component in components]
    return dict(polygons=map_boxes_to_clean(boxes,metadata,clean_size),scores=scores)


def complementarity(baseline,craft,valid_ids):
    baseline=set(map(str,baseline));craft=set(map(str,craft));valid_ids=set(map(str,valid_ids))
    require((baseline|craft)<=valid_ids,'Unknown GT IDs in complementary detections')
    return dict(both=len(baseline&craft),b1_only=len(baseline-craft),craft_only=len(craft-baseline),
                neither=len(valid_ids-(baseline|craft)))
