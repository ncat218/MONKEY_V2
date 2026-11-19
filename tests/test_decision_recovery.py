"""Recovery must reject mismatched caches and preserve the global review sample."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lowres.common import save_json
from rev4.decision_recovery import load_counts, merged_review
from rev4.decision_review import StratifiedReview


class RecoveryTests(unittest.TestCase):
    def test_reused_counts_reject_source_grid_and_content_changes(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'counts.npz'
            counts = np.zeros((2,2,1,5),dtype=np.int64)
            counts[:] = [2,4,3,1,1]
            settings = [dict(geometry='base',min_text_score=.4)]
            def store():
                np.savez(path,source_ids=np.asarray(['a','b']),
                         settings=np.asarray([json.dumps(settings[0])]),counts=counts)
            store()
            np.testing.assert_array_equal(load_counts(path,['a','b'],settings),counts)
            with self.assertRaises(ValueError):
                load_counts(path,['b','a'],settings)
            with self.assertRaises(ValueError):
                load_counts(path,['a','b'],[dict(geometry='different')])
            counts[0,0,0,0] = 10
            store()
            with self.assertRaises(ValueError):
                load_counts(path,['a','b'],settings)

    def test_chunk_merging_preserves_global_selection_and_population_weights(self):
        global_review = StratifiedReview()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            chunks = []
            for chunk_i in range(3):
                review = StratifiedReview()
                for i in range(7):
                    case = dict(case_id=f'{chunk_i}:{i}',condition='lr2',group='newer_only_fp',
                        height_bin='<4' if i < 4 else '4-<8',score=.6,
                        source_image_id=f'{chunk_i}:{i}',model='B1',ann_id=None,prediction_index=i)
                    review.add(case)
                    global_review.add(case)
                chunk = root/str(chunk_i)
                save_json(chunk/'selected_review.json',dict(cases=review.result()))
                save_json(chunk/'diagnosis_summary.json',dict(fp_proxies=[],
                    gt_transitions={'lr2':{'rescued':7}},fp_correspondence={'lr2':{'newer_only_fp':7}}))
                for name in ('fp_taxonomy_proxies.csv','gt_diagnosis.csv'):
                    (chunk/name).write_text('case_id\nexample\n',encoding='utf-8')
                chunks.append(chunk)
            cases,summary = merged_review(chunks,root/'merged')
            expected = global_review.result()
            self.assertEqual([c['case_id'] for c in cases],[c['case_id'] for c in expected])
            self.assertEqual([c['population_weight'] for c in cases],[c['population_weight'] for c in expected])
            self.assertEqual([c['inclusion_probability'] for c in cases],[c['inclusion_probability'] for c in expected])
            self.assertEqual(summary['gt_transitions']['lr2']['rescued'],21)
            self.assertEqual(summary['manual_review_status'],'PENDING')


if __name__ == '__main__':
    unittest.main()
