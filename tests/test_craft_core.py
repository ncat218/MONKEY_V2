import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from rev4.craft_core import complementarity,decode,map_boxes_to_clean,preprocess


class CraftTests(unittest.TestCase):
    def test_odd_dimension_coordinates_use_actual_resize_not_nominal_ratio(self):
        image=np.zeros((127,333,3),dtype=np.uint8)
        tensor,meta=preprocess(image)
        self.assertEqual(tensor.shape[0],3)
        self.assertTrue(all(v%32==0 for v in tensor.shape[1:]))
        rh,rw=meta['resized_hw']
        self.assertNotEqual(meta['actual_scale_xy'][0],meta['actual_scale_xy'][1])
        box=np.asarray([[0,0],[rw/2,0],[rw/2,rh/2],[0,rh/2]])
        result=np.asarray(map_boxes_to_clean([box],meta,[666,254])).reshape(4,2)
        np.testing.assert_allclose(result,[[0,0],[666,0],[666,254],[0,254]])

    def test_affinity_connects_characters_and_scores_satisfy_evaluator(self):
        region=np.zeros((48,64),np.float32)
        region[10:16,10:16]=.85;region[10:16,22:28]=1.2
        affinity=np.zeros_like(region)
        metadata=dict(resized_hw=[96,128])
        apart=decode(region,affinity,metadata,[128,96])
        self.assertEqual(len(apart['polygons']),2)
        affinity[11:15,15:23]=.8
        together=decode(region,affinity,metadata,[128,96])
        self.assertEqual(len(together['polygons']),1)
        self.assertTrue(all(0<=v<=1 for v in together['scores']))

    def test_gt_complementarity_is_not_an_ensemble_metric(self):
        self.assertEqual(complementarity(['1','2'],['2','3'],['1','2','3','4']),
            dict(both=1,b1_only=1,craft_only=1,neither=1))
        with self.assertRaises(ValueError):complementarity(['unknown'],[],['1'])

    def test_pinned_vendor_hashes(self):
        root=Path(__file__).resolve().parents[1]/'rev4/craft_vendor'
        provenance=json.loads((root/'provenance.json').read_text())
        for name,metadata in provenance['files'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),metadata['vendored_sha256'])


if __name__=='__main__':unittest.main()
