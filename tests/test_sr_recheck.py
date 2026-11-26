import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from lowres.common import sha256
from rev4.sr_plan import BRANCHES, PAIRS, RECIPE
from rev4.sr_recheck import completed_b1_detector, resized, validate_record, bootstrap_group


class RepeatRGB(torch.nn.Module):
    def forward(self, value):
        return value.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


class SRTests(unittest.TestCase):
    def test_completed_b1_is_loaded_instead_of_original_b0(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);b1=root/'b1.pth';b0=root/'b0.pth'
            b1.write_bytes(b'completed B1');b0.write_bytes(b'original B0')
            a=SimpleNamespace(checkpoint=b0,b1_checkpoint=b1,b1_digest=sha256(b1))
            detector=SimpleNamespace(model=torch.nn.Linear(2,2));native=SimpleNamespace()
            with patch('rev4.stage_a.detector_for',return_value=(detector,native,None,None)) as factory:
                result,processor=completed_b1_detector(a)
            self.assertEqual(factory.call_args.args[0].checkpoint,b1)
            self.assertEqual(a.checkpoint,b0)
            self.assertFalse(result.model.training)
            self.assertTrue(all(not p.requires_grad for p in result.model.parameters()))
            self.assertEqual((processor.mask_thr,processor.min_text_score),(.4,.6))
            b1.write_bytes(b'wrong weights')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):completed_b1_detector(a)

    def test_tiled_restore_stitches_odd_dimensions_without_changes(self):
        from rev4.edsr import restore
        pixels=np.random.default_rng(42).integers(0,256,(39,53,3),dtype=np.uint8)
        actual=restore(RepeatRGB(),pixels,'cpu',tile=16,halo=7)
        expected=pixels.repeat(2,axis=0).repeat(2,axis=1)
        np.testing.assert_array_equal(actual,expected)

    def test_upscaled_canvas_is_twice_native_size_not_clean_size(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'native.png'
            pixels=np.random.default_rng(1).integers(0,256,(15,22,3),dtype=np.uint8)
            Image.fromarray(pixels).save(source)
            original=dict(sample_id='lr2::one',source_image_id='one',condition='lr2',path=str(source),
                          size=[22,15],image_sha256=sha256(source))
            for branch in ('bicubic_x2','edsr_x2'):
                with patch.dict('os.environ',{'SLURM_TMPDIR':str(root)}):
                    with resized(SimpleNamespace(),original,branch,RepeatRGB(),'cpu') as (task,elapsed):
                        self.assertEqual(task['size'],[44,30])
                        self.assertNotEqual(task['size'],[45,31])
                        self.assertEqual(task['condition'],'lr2')
                        self.assertEqual(sha256(task['path']),task['image_sha256'])
                        self.assertGreaterEqual(elapsed,0)
                        with Image.open(task['path']) as result:
                            self.assertEqual(result.mode,'RGB')
                            self.assertEqual(result.size,(44,30))
                self.assertFalse(Path(task['path']).exists())

    def fixture(self):
        task=dict(sample_id='lr4::one',source_image_id='one',condition='lr4')
        counts=[1,2,3];bins=[[1,1],[1,0],[0,0],[0,0]]
        baseline=dict(image_sha256='source',counts=[counts]*6,bins=[bins]*6)
        branches={branch:dict(counts=copy.deepcopy([counts]*2),bins=copy.deepcopy([bins]*2),
                             matched=[['word'],['word']],size=[22,15] if branch=='native' else [44,30],
                             image_sha256='source' if branch=='native' else branch) for branch in BRANCHES}
        record=dict(**task,identity_sha256='identity',input_sha256='source',branches=branches)
        return task,baseline,record

    def test_cache_validation_handles_stress_tasks_and_rejects_changed_results(self):
        task,baseline,record=self.fixture()
        validate_record(record,task,'identity',baseline,[0,4])
        for kind in ('identity','native_counts','bins','missing_branch','size'):
            value=copy.deepcopy(record)
            if kind=='identity':value['identity_sha256']='other'
            elif kind=='native_counts':value['branches']['native']['counts'][0][2]+=1
            elif kind=='bins':value['branches']['edsr_x2']['bins'][0][0][1]=0
            elif kind=='missing_branch':value['branches'].pop('bicubic_x2')
            else:value['branches']['edsr_x2']['size']=[45,31]
            with self.assertRaises(ValueError):validate_record(value,task,'identity',baseline,[0,4])

    def test_bootstrap_can_find_small_text_gain_even_with_lower_hmean_and_more_fp(self):
        before=np.array([[1,3,2,1,0,1,1,1,0,0,0]]*3)
        after=np.array([[1,3,3,1,1,1,0,1,0,0,0]]*3)
        result=bootstrap_group(dict(native=before,bicubic_x2=before,edsr_x2=after),100)
        effect=result['edsr_x2_minus_native']
        self.assertLess(effect['hmean_pp']['ci95'][1],0)
        self.assertEqual(effect['recall_lt4_pp']['delta'],100)
        self.assertEqual(effect['recall_lt4_pp']['ci95'],[100,100])
        self.assertEqual(effect['fp_count']['delta'],3)
        with self.assertRaises(ValueError):
            bootstrap_group(dict(native=before,bicubic_x2=before,edsr_x2=after[:2]),10)

    @unittest.skipUnless(importlib.util.find_spec('shapely'),'GT geometry integration runs on HPC')
    def test_tiny_bins_use_original_native_lr2_height_not_sr_canvas(self):
        from rev4.decision_core import CandidateMatcher
        from shapely.geometry import box,mapping
        gt=mapping(box(1,1,11,9))
        row=dict(clean_size=[45,31],roi_clean=mapping(box(0,0,45,31)),
                 gt_clean=dict(valid=[dict(ann_id='word',geometry=gt)],ignored=[]))
        prediction=dict(polygons=[[1,1,11,1,11,9,1,9]],scores=[.9])
        policy=dict(match_iou=.5,min_prediction_roi_fraction=.5,ignore_prediction_overlap=.5)
        result=CandidateMatcher(prediction,[.9],row,policy).score(.6)
        self.assertEqual(result['bins'],[[1,1],[0,0],[0,0],[0,0]])

    def test_launcher_identity_survives_json_roundtrip(self):
        import submit_edsr_recheck as submit
        a=submit.light()
        self.assertEqual(json.loads(json.dumps(a.launch_identity)),a.launch_identity)
        self.assertTrue(a.launch_identity['no_training'])
        self.assertEqual(RECIPE['primary_pair'],[.4,.6])
        self.assertIn('--worker',submit.command(a)[-1])


if __name__=='__main__':unittest.main()
