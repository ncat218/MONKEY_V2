"""Verify explicit mask cutoff, frozen-kernel equivalence and source count guards."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box, mapping

import lowres.inference as legacy
from rev4.fixed_diagnostic_core import (MASK, SCORE, assert_reference,
                                        instance_transitions, kernel_at_threshold)
from rev4.fixed_review import HEIGHT, WIDTH, combine, render, valid_panel


def offset(points,distance):
    polygon=Polygon(points).buffer(distance,join_style=2)
    return np.asarray(polygon.exterior.coords[:-1]) if not polygon.is_empty else np.asarray([])


class FixedDiagnosticTests(unittest.TestCase):
    def test_kernel_geometry_matches_frozen_helper_when_cutoff_is_030(self):
        rng=np.random.default_rng(17)
        probability=rng.uniform(0,1,(32,32)).astype(np.float32)
        task=dict(size=[8,8])
        row=dict(clean_size=[16,16],roi_clean=mapping(box(0,0,16,16)),
            gt_clean=dict(valid=[dict(ann_id='a',geometry=mapping(box(2,2,13,10))),
                                 dict(ann_id='b',geometry=mapping(box(1,1,2,2)))],
                          ignored=[dict(geometry=mapping(box(10,2,12,6)))]))
        args=(probability,(16,16),np.array([2.,2.]),task,row,task['size'],.4,offset)
        expected=legacy.kernel_evidence(*args)
        actual=kernel_at_threshold(*args,mask_threshold=.30)
        self.assertEqual(expected[0],actual[0])
        np.testing.assert_array_equal(expected[1],actual[1])
        np.testing.assert_array_equal(expected[2],actual[2])
        self.assertEqual(legacy.MASK_THRESHOLD,.30)

    def test_fixed_040_cutoff_changes_coverage_without_mutating_legacy(self):
        probability=np.full((16,16),.35,dtype=np.float32)
        probability[:,8:]=.8
        probability[0,:3]=[np.nextafter(np.float32(.4),np.float32(0)),.4,
                            np.nextafter(np.float32(.4),np.float32(1))]
        task=dict(size=[16,16])
        row=dict(clean_size=[16,16],roi_clean=mapping(box(0,0,16,16)),
                 gt_clean=dict(valid=[dict(ann_id='a',geometry=mapping(box(2,2,13,12)))],ignored=[]))
        args=(probability,(16,16),np.array([1.,1.]),task,row,task['size'],.4,offset)
        before=legacy.kernel_evidence(*args)
        after=kernel_at_threshold(*args)
        self.assertGreater(before[0]['a']['coverage'],after[0]['a']['coverage'])
        np.testing.assert_array_equal(after[2],probability>MASK)
        self.assertEqual(after[2][0,:3].tolist(),[False,False,True])
        self.assertEqual(legacy.MASK_THRESHOLD,.30)
        self.assertEqual((MASK,SCORE),(.40,.60))

    def test_reference_check_covers_tiny_tp_not_only_total_tp(self):
        result=dict(counts=[1,2,3,1],instances={'a':dict(tp=True),'b':dict(tp=False)})
        attributes={'a':dict(height_lr2=3.),'b':dict(height_lr2=8.)}
        assert_reference(result,attributes,[1,2,3,1,1])
        with self.assertRaisesRegex(ValueError,'differs'):
            assert_reference(result,attributes,[1,2,3,1,0])

    def test_postprocess_transitions_reconcile_totals_and_reject_different_ids(self):
        before=dict(counts=[1,3,1,1],instances={'a':dict(tp=True),'b':dict(tp=False),'c':dict(tp=False)})
        after=dict(counts=[1,3,2,1],instances={'a':dict(tp=False),'b':dict(tp=True),'c':dict(tp=False)})
        self.assertEqual(instance_transitions(before,after),dict(regressed=1,rescued=1,persistent_fn=1))
        after['instances']['d']=after['instances'].pop('c')
        with self.assertRaisesRegex(ValueError,'GT IDs differ'):
            instance_transitions(before,after)

    def test_review_cards_render_both_native_and_clean_context_and_detect_partial_files(self):
        class Map:
            def __getitem__(self,index): return self
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.full((32,32),.8,dtype=np.float32)
        with TemporaryDirectory() as temp:
            root=Path(temp)
            Image.new('RGB',(32,32),'gray').save(root/'clean.png')
            Image.new('RGB',(16,16),'gray').save(root/'lr.png')
            task=dict(path=str(root/'lr.png'),size=[16,16],source_image_id='x',condition='lr2')
            clean_task=dict(path=str(root/'clean.png'))
            row=dict(clean_size=[32,32],gt_clean=dict(valid=[dict(ann_id='a',geometry=mapping(box(4,4,18,18)))],
                                                   ignored=[dict(geometry=mapping(box(20,20,26,26)))]))
            case=dict(source_image_id='x',condition='lr2',bounds=[4,4,18,18],ann_id='a',
                      case_id='x:a',group='regressed',panel_index=0,image_pair='examples/0000.png')
            prediction=dict(polygons=[[4,4,18,4,18,18,4,18]],scores=[.8])
            options=SimpleNamespace(output=root)
            for name in ('B-Clean','B1'):
                render(options,dict(name=name),task,clean_task,row,Map(),np.asarray([2.,2.]),prediction,[case])
                self.assertTrue(valid_panel(root/f'review/panels/0000_{name}.png'))
            combine(options,[case])
            with Image.open(root/'review/examples/0000.png') as image:
                self.assertEqual(image.size,(2*WIDTH,HEIGHT))
            broken=root/'broken.png'
            broken.write_bytes(b'partial PNG')
            self.assertFalse(valid_panel(broken))


if __name__=='__main__':
    unittest.main()
