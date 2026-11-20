import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lowres.common import read_json
from rev4.durable_io import save_json,storage_probe,write_bytes
from rev4.latent_recovery import verified_validation
from types import SimpleNamespace
from lowres.common import sha256


class DurableTests(unittest.TestCase):
    def test_fsync_failure_preserves_last_good_file(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'cache.json'
            save_json(p,dict(version=1))
            with patch('rev4.durable_io.os.fsync',side_effect=OSError('storage failure')):
                with self.assertRaises(OSError):save_json(p,dict(version=2))
            self.assertEqual(read_json(p),dict(version=1))
            self.assertEqual(list(p.parent.glob('*.tmp')),[])

    def test_zero_write_is_rejected_before_target_replace(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'cache.json';save_json(p,dict(version=1))
            original=Path.read_bytes
            def read(path):
                return b'' if path.suffix=='.tmp' else original(path)
            with patch.object(Path,'read_bytes',read):
                with self.assertRaisesRegex(OSError,'read-back'):save_json(p,dict(version=2))
            self.assertEqual(read_json(p),dict(version=1))
            with self.assertRaises(ValueError):write_bytes(p,b'')

    def test_storage_probe_and_roundtrip(self):
        with TemporaryDirectory() as directory:
            storage_probe(directory)
            self.assertEqual(read_json(Path(directory)/'storage_probe.json')['status'],'PASS')
            self.assertFalse((Path(directory)/'storage_probe_payload.json').exists())

    def test_reused_validation_requires_complete_identity_and_size_counts(self):
        task=dict(sample_id='clean::one',source_image_id='one',condition='clean')
        bins=[[2,1],[0,0],[0,0],[0,0]]
        value=dict(identity={'frozen':True},arm='head',tasks=[dict(**task,counts=[[1,2,3,1]]*2,bins=[bins,bins])])
        verified_validation(value,value['identity'],[task])
        broken=copy.deepcopy(value);broken['tasks'][0]['bins'][0][0][1]=0
        with self.assertRaises(ValueError):verified_validation(broken,value['identity'],[task])
        with self.assertRaises(ValueError):verified_validation(value,{'other':True},[task])
        with self.assertRaises(ValueError):verified_validation(value,value['identity'],[task,task])

    def test_finished_arm_is_not_trained_again_and_changed_checkpoint_stops(self):
        from rev4 import latent_trial
        with TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'original.pth'
            checkpoint.write_bytes(b'fixture completed checkpoint')
            ctx=SimpleNamespace(output=root,trial_identity={'frozen':'same'})
            save_json(root/'plain/trained.json',dict(identity=ctx.trial_identity,
                      checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint)))
            with (patch.object(latent_trial,'runner_for',side_effect=AssertionError('must not train')),
                  patch.dict('sys.modules',{'rev4.latent_plugins':SimpleNamespace(LatentAuditHook=object)})):
                latent_trial.train(ctx,'plain')
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):latent_trial.train(ctx,'plain')


if __name__=='__main__':unittest.main()
