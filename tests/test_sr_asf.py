import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lowres.common import sha256
from rev4.sr_asf import completed_asf_detector,same_inputs,paired_effects
from rev4.sr_asf_plan import CELLS,BASELINE_JOB


class ComboTests(unittest.TestCase):
    def test_loader_uses_completed_asf_weights_and_architecture_not_b1(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);asf=root/'asf.pth';b1=root/'b1.pth'
            asf.write_bytes(b'completed ASF');b1.write_bytes(b'B1')
            a=SimpleNamespace(output=root,asf_checkpoint=asf,asf_digest=sha256(asf),checkpoint=b1)
            model=torch.nn.Linear(2,2);model.neck=SimpleNamespace(mode='asf',asf_cfg={'attention_type':'ScaleChannelSpatial'})
            detector=SimpleNamespace(model=model);native=SimpleNamespace()
            with patch('rev4.sr_asf.configuration',return_value=SimpleNamespace(pretty_text='model = dict()')) as cfg, \
                 patch('rev4.stage_a.detector_for',return_value=(detector,native,None,None)) as factory:
                completed_asf_detector(a)
            self.assertEqual(cfg.call_args.args[1],'asf')
            self.assertEqual(factory.call_args.args[0].checkpoint,asf)
            self.assertEqual(a.checkpoint,b1)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertEqual((native.mask_thr,native.min_text_score),(.4,.6))
            asf.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'ASF checkpoint changed'):completed_asf_detector(a)

    def test_same_sources_are_insufficient_when_restoration_pixels_differ(self):
        value=dict(sample_id='lr2::one',input_sha256='input',branches={
            name:dict(size=[20,16],image_sha256=name) for name in ('native','bicubic_x2','edsr_x2')})
        same_inputs(value,copy.deepcopy(value))
        changed=copy.deepcopy(value);changed['branches']['edsr_x2']['image_sha256']='different'
        with self.assertRaisesRegex(ValueError,'pixels differ'):same_inputs(changed,value)
        changed=copy.deepcopy(value);changed['sample_id']='lr2::other'
        with self.assertRaises(ValueError):same_inputs(changed,value)

    def test_paired_factorial_interaction_is_computed_not_added(self):
        native=np.asarray([[1,3,2,1,0,1,1,1,0,0,0]]*3)
        sr=np.asarray([[2,3,3,1,1,1,1,1,0,0,0]]*3)
        asf=np.asarray([[1,3,2,1,1,1,0,1,0,0,0]]*3)
        combo=np.asarray([[3,3,3,1,1,1,1,1,1,0,0]]*3)
        cells={name:native.copy() for name in CELLS}
        cells.update({'B1/edsr_x2':sr,'ASF/native':asf,'ASF/edsr_x2':combo})
        effects=paired_effects(cells,repetitions=100)
        self.assertGreater(effects['ASF/edsr_x2_minus_B1/edsr_x2']['hmean_pp']['ci95'][0],0)
        self.assertAlmostEqual(effects['interaction']['hmean_pp']['delta'],100/3)
        self.assertEqual(effects['interaction']['recall_lt4_pp']['ci95'],[-100,-100])
        del cells['B1/bicubic_x2']
        with self.assertRaises(ValueError):paired_effects(cells,10)

    def test_dependency_waits_for_controls_and_rejects_parent_failure(self):
        import submit_edsr_asf as submit
        for status,expected in [('RUNNING|0:0\n',BASELINE_JOB),('COMPLETED|0:0\n',None)]:
            with patch.object(submit.subprocess,'check_output',return_value=status):
                self.assertEqual(submit.dependency(),expected)
        with patch.object(submit.subprocess,'check_output',return_value='FAILED|1:0\n'):
            with self.assertRaisesRegex(ValueError,'parent failed'):submit.dependency()

    def test_json_identity_and_dependency_argv_do_not_modify_running_baseline(self):
        import submit_edsr_asf as submit
        import submit_edsr_recheck as baseline
        before=copy.deepcopy(baseline.light().launch_identity)
        a=submit.light()
        self.assertEqual(json.loads(json.dumps(a.launch_identity)),a.launch_identity)
        self.assertEqual(before,baseline.light().launch_identity)
        self.assertIn('--dependency=afterok:'+BASELINE_JOB,submit.command(a,BASELINE_JOB))
        self.assertEqual(a.output.name,'v1')
        self.assertEqual(a.output.parent.name,'edsr_asf')


if __name__=='__main__':unittest.main()
