"""Synthetic checks for the locked B1 training recipe."""
from itertools import islice
from unittest import TestCase

from PIL import Image

from rev4.b1_training import choose_checkpoint
from rev4.training_data import balanced_indices, lr2_dimensions, native_lr2, scale_instance


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
