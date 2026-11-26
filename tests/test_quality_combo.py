import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box,mapping

from lowres.common import read_json
from rev4.quality_combo import frozen_model,evaluate_pool,assert_reference,validate,fold,same_candidates
from rev4.quality_combo_plan import CELLS,ASSETS,PARENT_JOB
from rev4.quality_combo_report import effects,mean_h_effects

ROOT=Path(__file__).resolve().parents[1]


class FrozenQualityTests(unittest.TestCase):
    def test_exact_published_model_and_feature_predictions(self):
        from mixab.boxquality_devA_janek import full_scores
        m=frozen_model(ROOT)
        self.assertEqual(m['cut_full'],.3315908613227198)
        self.assertEqual(m['cut_raw'],.5976406401908236)
        z=m['mean'][None,:]
        expected=1/(1+np.exp(-m['beta'][0]))
        self.assertAlmostEqual(full_scores(m,z.tolist())[0],expected)
        self.assertEqual(full_scores(m,[]),[])
        with TemporaryDirectory() as d:
            root=Path(d);(root/ASSETS).mkdir(parents=True)
            (root/ASSETS/'provenance.json').write_text((ROOT/ASSETS/'provenance.json').read_text())
            with self.assertRaises((ValueError,FileNotFoundError)):frozen_model(root)

    def test_raw_float_precision_floor_and_quality_filter_preserve_matching(self):
        row=dict(clean_size=[100,100],roi_clean=mapping(box(0,0,100,100)),
            gt_clean=dict(valid=[dict(ann_id='word',geometry=mapping(box(10,10,30,16)),original_height=6)],ignored=[]))
        policy=read_json(ROOT/'docs/handoff/textvqa_evaluation_protocol.json')
        pred=dict(polygons=[[10,10,30,10,30,16,10,16],[60,60,70,60,70,70,60,70]],scores=[.6,.9])
        raw=[.59999999999,.9]
        # The lower-raw true word can be retained by Q; Q never changes geometry or matching scores.
        m=dict(cut_raw=.5976406401908236,cut_full=.3315908613227198)
        with patch('mixab.boxquality_devA_janek.full_scores',return_value=[.9,.1]):
            results=evaluate_pool(pred,raw,[[0.]*13]*2,m,row,policy,official=True)
        self.assertEqual(results['F']['counts'],[0,1,1])
        self.assertEqual(results['RAW']['counts'],[1,1,2])
        self.assertEqual(results['Q']['counts'],[1,1,1])
        self.assertEqual(results['Q']['bins'][0],[1,1])
        with patch('mixab.boxquality_devA_janek.full_scores',return_value=[.99]) as scorer:
            result=evaluate_pool(pred,[.19,.9],[[0.]*13]*2,m,row,policy)
        self.assertEqual(len(scorer.call_args.args[1]),1)
        self.assertEqual(result['Q']['counts'],[0,1,1])
        with self.assertRaisesRegex(ValueError,'feature vector'):
            evaluate_pool(pred,raw,[[float('nan')]*13]*2,m,row,policy)

    def test_empty_pool_retains_gt_and_official_parity(self):
        row=dict(clean_size=[100,100],roi_clean=mapping(box(0,0,100,100)),
            gt_clean=dict(valid=[dict(ann_id='word',geometry=mapping(box(10,10,30,16)),original_height=6)],ignored=[]))
        policy=read_json(ROOT/'docs/handoff/textvqa_evaluation_protocol.json')
        result=evaluate_pool(dict(polygons=[],scores=[]),[],[],frozen_model(ROOT),row,policy,True)
        self.assertTrue(all(v['counts']==[0,1,0] for v in result.values()))

    def test_cache_rejects_changed_parent_pixels_or_false_native_counts(self):
        r=dict(counts=[1,1,2],bins=[[1,1],[0,0],[0,0],[0,0]],matched_ann_ids=['word'])
        ref=dict(counts=[[0,1,0],[1,1,2]],bins=[[[1,0],[0,0],[0,0],[0,0]],r['bins']],matched=[[],['word']])
        assert_reference(r,ref)
        p=dict(sample_id='lr2::one',input_sha256='input',branches={
            b:dict(ref,size=[10,10],image_sha256=b) for b in ('native','bicubic_x2','edsr_x2')})
        cache=dict(identity_sha256='id',parent_sha256='parent',sample_id='lr2::one',input_sha256='input',
            branches={b:dict(size=[10,10],image_sha256=b,feature_capture_exact=True,
                arms={n:copy.deepcopy(r) for n in ('F','RAW','Q')}) for b in p['branches']})
        validate(cache,p,'parent','id')
        changed=copy.deepcopy(cache);changed['branches']['edsr_x2']['image_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'bytes'):validate(changed,p,'parent','id')
        changed=copy.deepcopy(cache);changed['branches']['native']['arms']['F']['counts'][0]=0
        with self.assertRaisesRegex(ValueError,'control differs'):validate(changed,p,'parent','id')

    def test_paired_interactions_do_not_add_separate_method_gains(self):
        base=np.asarray([[1,3,2,1,1,1,0,1,0,0,0]]*5)
        higher=np.asarray([[2,3,2,1,1,1,1,1,0,0,0]]*5)
        matrices={k:base.copy() for k in CELLS}
        matrices['ASF/native/Q']=higher
        out=effects(matrices,100)
        self.assertAlmostEqual(out['quality_x_asf_native']['hmean_pp']['delta'],40.)
        self.assertEqual(out['quality_x_asf_native']['hmean_pp']['ci95'],[40.,40.])
        mean=mean_h_effects(matrices,matrices,100)
        self.assertAlmostEqual(mean['ASF/native/Q minus B1/native/F']['delta'],40.)
        del matrices['B1/native/F']
        with self.assertRaisesRegex(ValueError,'Incomplete'):effects(matrices,10)

    def test_condition_mean_is_mean_of_micro_metrics_with_paired_sources(self):
        base=np.asarray([[1,3,2,1,1,1,0,1,0,0,0]]*3)
        clean={k:base.copy() for k in CELLS};lr2=copy.deepcopy(clean)
        lr2['B1/native/Q'][:,0]=2;lr2['B1/native/Q'][:,6]=1
        r=mean_h_effects(clean,lr2,100)['B1/native/Q minus B1/native/F']
        self.assertAlmostEqual(r['delta'],20.)
        self.assertEqual(r['ci95'],[20.,20.])

    def test_capture_equality_includes_packed_scores_and_coordinates(self):
        pred=dict(polygons=[[0.,0.,1.,0.,1.,1.,0.,1.]],scores=[.6])
        self.assertTrue(same_candidates((pred,[.6]),(copy.deepcopy(pred),[.6])))
        changed=copy.deepcopy(pred);changed['polygons'][0][0]=.1
        self.assertFalse(same_candidates((pred,[.6]),(changed,[.6])))
        self.assertIn(fold('a'),('A','B'))

    def test_successor_dependency_and_parent_sources_are_unchanged(self):
        import submit_quality_combo as submit
        import submit_edsr_asf as parent
        before=parent.light().launch_identity
        a=submit.light()
        self.assertEqual(before,parent.light().launch_identity)
        self.assertTrue(all(a.launch_identity['source'][k]==v for k,v in before['source'].items()))
        for state,expected in [('RUNNING|0:0\n',PARENT_JOB),('PENDING|0:0\n',PARENT_JOB),('COMPLETED|0:0\n',None)]:
            with patch.object(submit.subprocess,'check_output',return_value=state):
                self.assertEqual(submit.dependency(),expected)
        with patch.object(submit.subprocess,'check_output',return_value='FAILED|1:0\n'):
            with self.assertRaisesRegex(ValueError,'Parent failed'):submit.dependency()
        cmd=submit.command(a,PARENT_JOB)
        self.assertIn('--dependency=afterok:'+PARENT_JOB,cmd)
        self.assertNotIn('scancel',' '.join(cmd))
        self.assertEqual(a.output.parent.name,'tri_quality_combo')


if __name__=='__main__':unittest.main()
