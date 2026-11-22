import copy
import errno
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from lowres.common import read_json, sha256
from rev4 import fusion_recovery as recovery
from rev4.durable_io import save_json
from rev4.fusion_evaluation import materialize, variant
from rev4.fusion_plan import PAIRS, RECIPE
from rev4.local_scratch import local_directory


class ScratchTests(unittest.TestCase):
    def test_cleanup_retries_transient_failure_and_preserves_real_errors(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {'SLURM_TMPDIR': directory}):
            temporary = SimpleNamespace(name=str(Path(directory)/'one'),
                cleanup=Mock(side_effect=[OSError(errno.ENOTEMPTY, 'race'), None]))
            with patch('rev4.local_scratch.tempfile.TemporaryDirectory', return_value=temporary), \
                    patch('rev4.local_scratch.time.sleep') as sleep:
                with local_directory('test-') as path:
                    self.assertEqual(path.parent, Path(directory))
                self.assertEqual(temporary.cleanup.call_count, 2)
                sleep.assert_called_once_with(.05)
            temporary.cleanup = Mock(side_effect=PermissionError(errno.EACCES, 'denied'))
            with patch('rev4.local_scratch.tempfile.TemporaryDirectory', return_value=temporary):
                with self.assertRaises(PermissionError):
                    with local_directory('test-'): pass
                self.assertEqual(temporary.cleanup.call_count, 1)
            temporary.cleanup = Mock(side_effect=OSError(errno.ENOTEMPTY, 'persistent'))
            with patch('rev4.local_scratch.tempfile.TemporaryDirectory', return_value=temporary), \
                    patch('rev4.local_scratch.time.sleep'):
                with self.assertRaises(OSError):
                    with local_directory('test-'): pass
                self.assertEqual(temporary.cleanup.call_count, 4)

    def test_all_corruptions_use_node_local_storage_without_changing_encoded_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root/'clean.png'
            image = Image.new('RGB', (45, 31), (48, 112, 183)); image.save(source)
            node = root/'node'; node.mkdir()
            a = SimpleNamespace(clean_root=root, output=root/'shared')
            row = dict(clean_relpath=source.name, clean_sha256=sha256(source), clean_size=[45, 31])
            with patch.dict(os.environ, {'SLURM_TMPDIR': str(node)}):
                for condition in ('lr4', 'lr8', 'blur1', 'blur2', 'jpeg30', 'jpeg10'):
                    expected, fmt = variant(image, condition)
                    expected = Image.fromarray(np.asarray(expected))
                    encoded = BytesIO()
                    options = dict(quality=int(condition[4:])) if fmt == 'JPEG' else {}
                    expected.save(encoded, format=fmt, **options)
                    with materialize(a, dict(source_image_id='one', condition=condition), row) as task:
                        path = Path(task['path'])
                        self.assertTrue(path.resolve().is_relative_to(node.resolve()))
                        self.assertEqual(path.read_bytes(), encoded.getvalue())
                    self.assertFalse(path.exists())
            self.assertFalse(a.output.exists())


class RecoveryTests(unittest.TestCase):
    def identities(self):
        old = dict(b1_checkpoint_sha256='b1'*32, bclean_checkpoint_sha256='bc'*32, data='frozen',
                   launch=dict(recipe=copy.deepcopy(RECIPE), source={
                       'submit_fusion_study.py':'old', 'rev4/fusion_evaluation.py':'old',
                       'rev4/fusion_training.py':'same'}))
        new = copy.deepcopy(old)
        new['launch']['source'].update({name:'new' for name in recovery.CHANGED_SOURCE | recovery.NEW_SOURCE})
        return old, new

    def test_identity_rejects_changed_training_recipe_or_data(self):
        old, new = self.identities(); recovery.validate_identity(old, new)
        for bad in ('training', 'recipe', 'data'):
            changed = copy.deepcopy(new)
            if bad == 'training': changed['launch']['source']['rev4/fusion_training.py'] = 'changed'
            elif bad == 'recipe': changed['launch']['recipe']['seed'] = 123
            else: changed['data'] = 'other'
            with self.assertRaises(ValueError): recovery.validate_identity(old, changed)

    def test_counts_reject_wrong_prefix_and_inconsistent_bins(self):
        parent, _ = self.identities()
        task = dict(sample_id='lr2::one', source_image_id='one', condition='lr2')
        row = dict(**task, counts=[[1, 2, 3] for _ in PAIRS],
                   bins=[[[2, 1], [0, 0], [0, 0], [0, 0]] for _ in PAIRS])
        value = dict(identity=parent, checkpoint_sha256='locked', tasks=[row])
        recovery.validate_counts(value, parent, [task], 1, 'locked')
        changed = copy.deepcopy(value); changed['tasks'][0]['sample_id'] = 'lr2::other'
        with self.assertRaises(ValueError): recovery.validate_counts(changed, parent, [task], 1, 'locked')
        changed = copy.deepcopy(value); changed['tasks'][0]['bins'][0][0][1] = 0
        with self.assertRaises(ValueError): recovery.validate_counts(changed, parent, [task], 1, 'locked')

    def test_restore_validates_hashes_and_keeps_prediction_progress_on_reentry(self):
        old, new = self.identities()
        with TemporaryDirectory() as directory:
            root = Path(directory); parent = root/'parent'; output = root/'recovery'
            tasks = [dict(sample_id=f'lr2::{i}', source_image_id=str(i), condition='lr2') for i in range(7524)]
            save_json(parent/'identity.json', old)
            save_json(parent/'submission/latest_job.json', dict(job_id='106090'))
            save_json(parent/'preflight.json', dict(status='PASS', identity=old))
            weights = {}; files = {}
            for arm in ('nearest', 'asf'):
                path = parent/f'arms/{arm}/weights/iter_20000.pth'
                path.parent.mkdir(parents=True); path.write_bytes(('completed '+arm).encode())
                weights[arm] = sha256(path)
                name = f'arms/{arm}/trained.json'
                save_json(parent/name, dict(identity=old, arm=arm, optimizer_updates=10000, micro_iterations=20000,
                    checkpoint=str(path), checkpoint_sha256=weights[arm], stream={'source':'same', 'augmentation':'same'}))
                files[name] = sha256(parent/name)
            for arm in ('B1', 'B-Clean', 'nearest', 'asf'):
                digest = weights.get(arm, old['b1_checkpoint_sha256' if arm=='B1' else 'bclean_checkpoint_sha256'])
                count = 5700 if arm=='asf' else 7524
                rows = [dict(**task, counts=[[1, 2, 3] for _ in PAIRS],
                             bins=[[[2, 1], [0, 0], [0, 0], [0, 0]] for _ in PAIRS]) for task in tasks[:count]]
                name = f'evaluation/{arm}/counts.json'
                save_json(parent/name, dict(identity=old, arm=arm, checkpoint_sha256=digest, tasks=rows))
                files[name] = sha256(parent/name)
            a = SimpleNamespace(recover_from=parent, output=output, trial_identity=new)
            with patch.object(recovery, 'PARENT_IDENTITY_SHA', sha256(parent/'identity.json')), \
                    patch.object(recovery, 'PARENT_FILES', files), patch.object(recovery, 'WEIGHT_SHA', weights), \
                    patch('rev4.fusion_evaluation.specification', return_value=({}, tasks, [])):
                recovery.restore(a)
                path = output/'evaluation/asf/counts.json'
                value = read_json(path); self.assertEqual(len(value['tasks']), 5700)
                extra = copy.deepcopy(value['tasks'][-1]); extra.update(tasks[5700]); value['tasks'].append(extra)
                save_json(path, value)
                recovery.restore(a)
                self.assertEqual(len(read_json(path)['tasks']), 5701)
                for name, digest in files.items(): self.assertEqual(sha256(parent/name), digest)
                checkpoint = parent/'arms/asf/weights/iter_20000.pth'
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'weights changed'): recovery.restore(a)

    def test_completed_arm_is_skipped_without_building_a_runner(self):
        from rev4 import fusion_training
        with TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root/'completed.pth'; checkpoint.write_bytes(b'complete')
            a = SimpleNamespace(output=root, trial_identity={'frozen':True})
            save_json(root/'arms/nearest/trained.json', dict(identity=a.trial_identity,
                      checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint)))
            with patch.dict('sys.modules', {'rev4.fusion_plugins':SimpleNamespace(FusionAuditHook=object)}), \
                    patch.object(fusion_training, 'runner_for', side_effect=AssertionError('must not retrain')):
                fusion_training.train(a, 'nearest')

    def test_launcher_propagates_recovery_to_allocation_and_children(self):
        import submit_fusion_study as submit
        a = submit.light(True)
        self.assertEqual(a.output.name, 'recovery_v2')
        self.assertIn('--recover', submit.command(a)[-1])
        with patch.object(submit.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            submit.child(a, 'evaluate', 'asf')
        self.assertIn('--recover', run.call_args.args[0])


if __name__ == '__main__': unittest.main()
