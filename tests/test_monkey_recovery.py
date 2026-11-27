import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from mmengine.structures import InstanceData
from mmocr.structures import TextDetDataSample
from mmocr.models.textdet.module_losses import DBModuleLoss

from rev4.prc2_plugins import Rev4PRC2DBHead,kernel_selection,teacher_kernels
from rev4.prc2_snapshot import Rev4PRC2SnapshotDBHead
from rev4.zoom_quality import predictions,fused,capture,assert_control


def sample(condition='lr2'):
    s=TextDetDataSample(metainfo=dict(img_shape=(128,128),condition=condition))
    s.gt_instances=InstanceData(polygons=[np.array([40,40,80,40,80,60,40,60],dtype=np.float32)],
                                ignored=torch.tensor([False]))
    return s


class MutationTests(unittest.TestCase):
    def test_installed_db_loss_reproduces_polygon_shift_and_lost_teacher_coverage(self):
        loss=DBModuleLoss();s=sample();before=s.gt_instances.polygons[0].copy()
        teacher=torch.zeros((128,128));teacher[40:61,40:81]=1
        def selected(s):
            k,_=teacher_kernels(loss,s,(128,128))
            return kernel_selection(teacher,torch.as_tensor(k,dtype=torch.long),torch.ones_like(teacher,dtype=torch.bool),.3,.3,4)[1]
        self.assertEqual(selected(s),1)
        loss.get_targets([s])
        self.assertFalse(np.array_equal(before,s.gt_instances.polygons[0]))
        self.assertEqual(selected(s),0)

    def test_snapshot_head_preserves_teacher_gt_while_student_loss_behavior_is_unchanged(self):
        db=DBModuleLoss()
        class MutatingLoss(nn.Module):
            def forward(self,outputs,samples):
                db.get_targets(samples)
                return {'loss_detection':outputs[0].sum()*0}
        head=object.__new__(Rev4PRC2SnapshotDBHead);nn.Module.__init__(head)
        head.module_loss=MutatingLoss();head.prc2={'weight':1.}
        student=torch.full((2,128,128),.5,requires_grad=True)
        head.forward=lambda x,samples,mode: (student,student,student,student)
        teacher=torch.zeros((128,128));teacher[40:61,40:81]=1
        def target(s):
            k,_=teacher_kernels(db,s,(128,128));labels=torch.as_tensor(k,dtype=torch.long)
            mask,selected,eligible=kernel_selection(teacher,labels,torch.ones_like(teacher,dtype=torch.bool),.3,.3,4)
            return teacher,mask,selected,eligible
        head.teacher_target=target
        samples=[sample('clean'),sample('lr2')]
        fixed=head.loss(None,samples)
        self.assertEqual(int(fixed['prc2_selected']),1)
        self.assertGreater(int(fixed['prc2_pixels']),0)
        fixed['loss_prc2'].backward()
        self.assertGreater(float(student.grad.abs().sum()),0)
        # Original detection target construction still mutates its own sample.
        self.assertFalse(np.array_equal(samples[1].gt_instances.polygons[0],sample().gt_instances.polygons[0]))
        old=Rev4PRC2DBHead.loss(head,None,[sample('clean'),sample('lr2')])
        self.assertEqual(int(old['prc2_selected']),0)


class ZoomTests(unittest.TestCase):
    def test_q_keeps_low_raw_box_through_fusion_and_preserves_nms_score(self):
        pred=dict(polygons=[[10,10,20,10,20,20,10,20]],scores=[.55])
        m={'cut_full':.3315908613227198}
        with patch('mixab.boxquality_devA_janek.full_scores',return_value=[.9]):
            pools=predictions(pred,[.55],[[0.]*13],m,.5,'global')
        self.assertEqual(pools['F'],[]);self.assertEqual(len(pools['Q']),1)
        self.assertEqual(pools['Q'][0]['score'],.55)
        self.assertEqual(len(fused(pools,[],[],(0,0,100,100),dict(tau_ref=8,margin_clean=1,nms_iou=.5),'Q')),1)

    def test_candidate_floor_and_nonfinite_features_are_enforced(self):
        pred=dict(polygons=[[0,0,1,0,1,1,0,1]],scores=[.19])
        with patch('mixab.boxquality_devA_janek.full_scores',return_value=[]) as scorer:
            p=predictions(pred,[.19],[[0.]*13],{'cut_full':.3},.5,'zoom')
        self.assertEqual(p,dict(F=[],Q=[]));self.assertEqual(scorer.call_args.args[1],[])
        with self.assertRaisesRegex(ValueError,'Invalid crop features'):
            predictions(pred,[.9],[[float('nan')]*13],{'cut_full':.3},.5,'zoom')

    def test_crop_features_are_extracted_before_shift_to_clean_coordinates(self):
        pred=dict(polygons=[[50,30,60,30,60,40,50,40]],scores=[.8]);raw=[.8]
        with patch('mixab.boxquality_janek.capture_with_features',return_value=(pred,raw,[[0.]*13])) as features, \
             patch('rev4.decision_core.capture_candidates',return_value=(pred,raw,{})), \
             patch('mixab.boxquality_devA_janek.full_scores',return_value=[.9]):
            pools,_=capture(None,None,None,np.array([.5,.5]),{'cut_full':.3},.5,'zoom',0,(100,50),np.array([2.,2.]))
        np.testing.assert_array_equal(features.call_args.args[3],[.5,.5])
        np.testing.assert_array_equal(pools['Q'][0]['points'][0],[300.,160.])

    def test_control_requires_global_quality_and_original_z0_parity(self):
        row=[1,2,2,1,1,1,0,0,0,0,0,1,0,0,0]
        r=dict(results={'global/F':row,'global/Q':row,'zoom_edsr/F':row},windows=[[0,0,10,10]])
        ref={'branches':{'native':{'arms':{s:dict(counts=row[:3],bins=[[1,1],[1,0],[0,0],[0,0]]) for s in ['F','Q']}}}}
        old=dict(windows={'guided':r['windows']},results={'guided_edsr':{'0.40/0.60':row}})
        assert_control(r,ref,old)
        changed=copy.deepcopy(r);changed['windows']=[[1,1,10,10]]
        with self.assertRaisesRegex(ValueError,'windows differ'):assert_control(changed,ref,old)


class RecoveryTests(unittest.TestCase):
    def test_recovery_source_does_not_change_old_launches_or_training_recipes(self):
        import submit_monkey_recovery as recover
        before=dict(quality=recover.quality.light().launch_identity,research=recover.research.light().launch_identity)
        a=recover.light()
        self.assertEqual(before,a.launch_identity['old_launches'])
        self.assertEqual(json.loads(json.dumps(a.launch_identity)),a.launch_identity)
        self.assertEqual(recover.PHASES[:3],('zoom_probe','B1_s43','BClean_s43'))
        self.assertIn('--signal=B:USR1@120',recover.command(a))

    def test_baseline_phase_only_evaluates_saved_checkpoints(self):
        import submit_monkey_recovery as recover
        ctx=SimpleNamespace()
        with patch.object(recover,'verified',return_value=ctx),patch.object(recover,'old_context',return_value=ctx), \
             patch('rev4.bundle_training.evaluate') as evaluate,patch('rev4.bundle_training.fit') as fit:
            recover.run_phase(None,'B1_s43')
        evaluate.assert_called_once_with(ctx,'B1_s43');fit.assert_not_called()

    def test_only_explicit_low_active_guard_is_a_scientific_stop(self):
        import submit_monkey_recovery as recover
        with TemporaryDirectory() as d:
            ctx=SimpleNamespace(output=Path(d),trial_identity={'fixed':'identity'})
            with patch.object(recover,'snapshot_context',return_value=ctx), \
                 patch('rev4.bundle_training.fit',side_effect=ValueError('PRC2 active in only 7/200 early microbatches; stopping')):
                recover.snapshot_phase(None,'prc2_snapshot_train')
            result=json.loads((Path(d)/'stopped.json').read_text())
            self.assertEqual(result['outcome'],'STOPPED_LOW_ACTIVE');self.assertFalse(result['thresholds_relaxed'])
            with patch.object(recover,'snapshot_context',return_value=ctx), \
                 patch('rev4.bundle_training.fit',side_effect=ValueError('checkpoint corrupted')):
                with self.assertRaisesRegex(ValueError,'checkpoint corrupted'):
                    recover.snapshot_phase(None,'prc2_snapshot_train')


if __name__=='__main__':unittest.main()
