import copy
import unittest

import torch
from torch import nn

from rev4.latent_modules import build_refiner,frozen_digest,load_base_state


class Toy(nn.Module):
    def __init__(self,kind='head'):
        super().__init__()
        self.backbone=nn.Conv2d(3,4,1)
        self.neck=nn.BatchNorm2d(4)
        self.det_head=nn.Conv2d(4,1,1)
        self.feature_adapter=build_refiner(kind)


class LatentTests(unittest.TestCase):
    def test_identity_shape_and_fp32_in_both_arms(self):
        x=torch.randn(2,256,7,9)
        for kind in ('head','plain','gated'):
            model=build_refiner(kind)
            self.assertTrue(torch.equal(model(x),x))
            self.assertEqual(model(x).dtype,torch.float32)

    def test_adapter_body_gradients_start_after_zero_output_learns(self):
        x=torch.randn(2,256,8,8)
        for kind in ('plain','gated'):
            model=build_refiner(kind)
            optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            model(x).square().mean().backward()
            self.assertGreater(float(model.project_out.weight.grad.abs().sum()),0)
            self.assertEqual(float(model.project_in.weight.grad.abs().sum()),0)
            optimizer.step();optimizer.zero_grad()
            model(x).square().mean().backward()
            self.assertGreater(float(model.project_in.weight.grad.abs().sum()),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_capacity_is_close_and_initialization_preserves_rng(self):
        torch.manual_seed(15);state=torch.random.get_rng_state().clone()
        plain=build_refiner('plain');gated=build_refiner('gated')
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        n=sum(p.numel() for p in plain.parameters());m=sum(p.numel() for p in gated.parameters())
        self.assertLess(abs(n-m)/n,.01)

    def test_base_load_allows_only_new_adapter_keys(self):
        base=Toy();state=copy.deepcopy(base.state_dict());model=Toy('gated')
        load_base_state(model,state)
        for key,value in state.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        broken=dict(state);broken.pop('backbone.weight')
        with self.assertRaises(ValueError):load_base_state(model,broken)
        broken=dict(state);broken['unknown.weight']=torch.ones(1)
        with self.assertRaises(ValueError):load_base_state(model,broken)

    def test_frozen_digest_includes_bn_buffers_and_excludes_head(self):
        model=Toy();before=frozen_digest(model)
        with torch.no_grad():model.det_head.weight.add_(1)
        self.assertEqual(frozen_digest(model),before)
        model.neck.running_mean.add_(1)
        self.assertNotEqual(frozen_digest(model),before)


if __name__=='__main__':unittest.main()
