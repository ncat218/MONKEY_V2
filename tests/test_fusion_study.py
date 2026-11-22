import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from rev4.fusion_modules import sampler,load_shared
from rev4.fusion_plan import RECIPE,ARMS,PAIRS
from rev4.fusion_evaluation import stress_ids,variant,metrics

try:
    from mmengine.config import Config
except ImportError:
    Config=None


class FusionTests(unittest.TestCase):
    def test_zero_offsets_match_bilinear_and_receive_gradients(self):
        for scale in (2,4,8):
            x=torch.randn(2,64,5,7);module=sampler(64,scale)
            actual=module(x);target=F.interpolate(x,scale_factor=scale,mode='bilinear',align_corners=False)
            torch.testing.assert_close(actual,target,atol=3e-6,rtol=2e-5)
            actual.square().mean().backward()
            self.assertGreater(float(module.offset.weight.grad.abs().sum()),0)
            self.assertTrue(torch.isfinite(module.offset.weight.grad).all())

    def test_shared_loading_rejects_missing_base_or_unexpected_tensors(self):
        class Model(nn.Module):
            def __init__(self,extra=False):
                super().__init__();self.neck=nn.Module();self.neck.base=nn.Linear(4,4)
                if extra:self.neck.asf_conv=nn.Linear(4,4)
        source=copy.deepcopy(Model().state_dict());model=Model(True)
        load_shared(model,source)
        broken=dict(source);broken.pop('neck.base.weight')
        with self.assertRaises(ValueError):load_shared(model,broken)
        broken=dict(source);broken['unknown']=torch.ones(1)
        with self.assertRaises(ValueError):load_shared(model,broken)

    def test_stress_selection_is_prediction_free_and_order_invariant(self):
        rows={str(i):{} for i in range(700)}
        selected=stress_ids(rows)
        self.assertEqual(len(selected),500)
        self.assertEqual(selected,stress_ids(dict(reversed(list(rows.items())))))
        self.assertEqual(len(set(selected)),500)

    def test_corruption_geometry_and_clean_pixels_are_preserved(self):
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(31,45,3),dtype=np.uint8))
        original=np.asarray(image).copy()
        for condition in ('lr4','lr8','blur1','blur2','jpeg30','jpeg10'):
            result,_=variant(image,condition)
            size=(45//int(condition[2:]),31//int(condition[2:])) if condition.startswith('lr') else image.size
            self.assertEqual(result.size,size)
            self.assertTrue(np.array_equal(np.asarray(image),original))

    def test_pinned_vendor_sources(self):
        root=Path(__file__).resolve().parents[1]/'rev4/dysample_vendor'
        provenance=json.loads((root/'provenance.json').read_text())
        self.assertEqual(provenance['commit'],'81a1de5caa95d55a0f5488425fa53ec7ef47f8f0')
        for name,entry in provenance['files'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),entry['vendored_sha256'])

    def test_failed_child_stops_the_controller(self):
        import submit_fusion_study
        a=SimpleNamespace(python=Path('python'))
        with patch.object(submit_fusion_study.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(ValueError,'stopped before next phase'):submit_fusion_study.child(a,'probe')

    @unittest.skipIf(Config is None,'MMEngine integration must run on HPC')
    def test_installed_fpnc_controls_and_sampling_shapes(self):
        from mmocr.models.textdet.necks import FPNC
        from rev4.fusion_plugins import StudyFPNC
        kwargs=dict(in_channels=[64,128,256,512],lateral_channels=256)
        reference=FPNC(**kwargs)
        nearest=StudyFPNC(mode='nearest',**kwargs);nearest.load_state_dict(reference.state_dict(),strict=True)
        features=[torch.randn(1,c,16//(2**i),24//(2**i)) for i,c in enumerate(kwargs['in_channels'])]
        self.assertTrue(torch.equal(reference(features),nearest(features)))
        bilinear=StudyFPNC(mode='bilinear',**kwargs);bilinear.load_state_dict(reference.state_dict(),strict=True)
        dynamic=StudyFPNC(mode='dysample',**kwargs)
        dynamic.load_state_dict(reference.state_dict(),strict=False)
        torch.testing.assert_close(dynamic(features),bilinear(features),atol=1e-5,rtol=1e-4)
        asf=StudyFPNC(mode='asf',asf_cfg=dict(attention_type='ScaleChannelSpatial'),**kwargs)
        value=asf(features)
        self.assertEqual(tuple(value.shape),(1,256,16,24))
        value.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in asf.parameters() if p.grad is not None))

    @unittest.skipIf(Config is None,'MMEngine integration must run on HPC')
    def test_equal_budgets_and_continuous_pilot_schedule(self):
        from rev4.fusion_training import configuration
        root=Path(__file__).resolve().parents[1]
        a=SimpleNamespace(model_config=root/'docs/handoff/inference_config.py',calibration_prepared=Path('prepared'),
                          output=Path('study'),checkpoint=Path('b0.pth'))
        normalized=[]
        for arm in ARMS:
            cfg=configuration(a,arm)
            self.assertEqual(cfg.train_cfg.max_iters,20000)
            self.assertEqual(cfg.param_scheduler[1]['end'],20000)
            self.assertEqual(cfg.custom_hooks,[dict(type='FusionAuditHook',pilot_updates=1000,accumulation=2)])
            self.assertIs(cfg.model.backbone['freeze'],True)
            self.assertEqual(cfg.model.neck.asf_cfg,dict(attention_type='ScaleChannelSpatial') if arm=='asf' else None)
            cfg=cfg.to_dict();cfg.pop('work_dir');cfg['model']['neck'].pop('mode');cfg['model']['neck'].pop('asf_cfg')
            normalized.append(cfg)
        self.assertTrue(all(v==normalized[0] for v in normalized))
        self.assertEqual(metrics([3,5,4])['hmean'],6/9)


if __name__=='__main__':unittest.main()
