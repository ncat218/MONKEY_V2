"""Synthetic checks for the locked B1 training recipe."""
from itertools import islice
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from lowres.common import sha256
from rev4 import training_data
from rev4.b1_training import choose_checkpoint
from rev4.training_data import balanced_indices, degraded_relative, lr2_dimensions, native_lr2, scale_instance
from submit_b1 import sbatch_args


class B1Tests(TestCase):
    def test_one_clean_and_one_lr2_per_microbatch(self):
        count = 7
        draws = list(islice(balanced_indices(count, 42), 4*count))
        self.assertEqual(draws, list(islice(balanced_indices(count, 42), 4*count)))
        for start in (0, 2*count):
            chunk = draws[start:start+2*count]
            self.assertEqual(sorted(chunk[::2]), list(range(count)))
            self.assertEqual(sorted(chunk[1::2]), list(range(count, 2*count)))

    def test_odd_image_scales_by_actual_width_and_height(self):
        image = Image.new('RGB', (15, 11), (85, 50, 180))
        dimensions = lr2_dimensions(image.size)
        self.assertEqual(dimensions, (7, 5))
        self.assertEqual(native_lr2(image).tobytes(),
                         image.resize(dimensions, Image.Resampling.BICUBIC).tobytes())
        original = dict(polygon=[0, 0, 15, 0, 15, 11, 0, 11], bbox=[0, 0, 15, 11],
                        bbox_label=0, ignore=True)
        lr2 = scale_instance(original, image.size, dimensions)
        self.assertEqual(lr2['bbox'], [0, 0, 7, 5])
        self.assertEqual(lr2['polygon'], [0, 0, 7, 0, 7, 5, 0, 5])
        self.assertTrue(lr2['ignore'])
        self.assertEqual(original['bbox'], [0, 0, 15, 11])

    def test_checkpoint_selection_uses_balanced_score_and_earlier_tie(self):
        early = dict(balanced_hmean=.64, identity=dict(updates=5_000))
        late = dict(balanced_hmean=.64, identity=dict(updates=10_000))
        self.assertIs(choose_checkpoint([late, early]), early)
        late['balanced_hmean'] = .641
        self.assertIs(choose_checkpoint([early, late]), late)

    def test_archived_train_is_usable_without_mixed_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archived = root/'backup.jsonl'
            prepared = root/'prepared.jsonl'
            archived.write_bytes(b'{"source_image_id":"fixture"}\n')
            inputs = SimpleNamespace(train_manifest=archived)
            with patch.object(training_data, 'TRAIN_SHA256', sha256(archived)):
                training_data.recover_train(inputs, prepared, root, None, None, None)
                self.assertEqual(prepared.read_bytes(), archived.read_bytes())
                archived.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'TRAIN SHA-256 mismatch'):
                    training_data.recover_train(inputs, prepared, root, None, None, None)

    def test_metadata_paths_preserve_frozen_relative_path(self):
        relative = 'main/images/realistic_mix/L2/train/0002c799b0cd7412.png'
        for recorded in ('/old/root/'+relative, 'C:\\old\\root\\'+relative.replace('/', '\\')):
            self.assertEqual(degraded_relative(recorded).as_posix(), relative)
        for recorded in ('/old/root/images/source.png', '/old/root/main/images', 'main/images/../source.png'):
            with self.assertRaisesRegex(ValueError, 'Unexpected recorded mixed path'):
                degraded_relative(recorded)

    def test_worker_receives_alternate_backup_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/'archived train.jsonl'
            manifest.touch()
            inputs = SimpleNamespace(s1=root/'s1', cuda=root/'cuda', python=root/'python',
                                     full_r1_root=root/'metadata backup', train_manifest=manifest)
            worker = shlex.split(sbatch_args(inputs)[-1].split('; exec ', 1)[1])
            self.assertEqual(worker[-4:], ['--full-r1-root', str(inputs.full_r1_root),
                                          '--train-manifest', str(manifest)])
