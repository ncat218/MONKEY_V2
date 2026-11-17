"""Checks for alignment, foreground validity and the actual distillation gradient."""
import unittest

import numpy as np
import torch

from rev4.feature_distillation import feature_mask, masked_feature_loss, teacher_canvas, text_mask


class FeatureDistillationTests(unittest.TestCase):
    def test_native_two_x_canvas_preserves_clean_pixels(self):
        clean = np.arange(8*12*3, dtype=np.uint8).reshape(8, 12, 3)
        image, valid = teacher_canvas(clean, (4, 6), np.array([[1., 0., 0.], [0., 1., 0.]]),
                                      (0, 0), (1., 1.), (4, 6))
        np.testing.assert_array_equal(image, clean)
        self.assertTrue(valid.all())

    def test_flip_and_crop_keep_same_field_of_view(self):
        clean = np.arange(8*12, dtype=np.uint8).reshape(8, 12)
        image, valid = teacher_canvas(clean, (4, 6), np.array([[-1., 0., 6.], [0., 1., 0.]]),
                                      (1, 1), (1., 1.), (2, 3))
        np.testing.assert_array_equal(image, clean[:, ::-1][2:6, 2:8])
        self.assertTrue(valid.all())

    def test_odd_sizes_rotation_and_outside_canvas(self):
        image, valid = teacher_canvas(np.ones((7, 9), dtype=np.uint8), (3, 4),
                                      np.array([[0., -1., 3.], [1., 0., 0.]]),
                                      (0, 0), (1., 1.), (5, 5))
        self.assertEqual(image.shape, (10, 10))
        self.assertEqual(valid.shape, (5, 5))
        self.assertTrue(valid[:4, :3].all())
        self.assertFalse(valid[4:, :].any())
        self.assertFalse(valid[:, 3:].any())

    def test_ignored_and_padding_are_excluded_from_features(self):
        valid = np.ones((8, 8), dtype=bool)
        valid[6:] = False
        polygons = [np.array([0, 0, 7, 0, 7, 7, 0, 7]), np.array([0, 0, 3, 0, 3, 3, 0, 3])]
        mask, allowed = text_mask((8, 8), polygons, [False, True], valid)
        self.assertFalse(mask[:4, :4].any())
        self.assertFalse(mask[6:].any())
        weights = feature_mask(torch.tensor(mask), torch.tensor(allowed), (2, 2))
        torch.testing.assert_close(weights, torch.tensor([[[[0., 1.], [0., 0.]]]]))

    def test_kd_detaches_teacher_and_only_trains_masked_student(self):
        student = torch.tensor([[[[1., 2.]], [[2., 1.]]]], requires_grad=True)
        teacher = torch.tensor([[[[2., 9., 1., 1.], [2., 9., 1., 1.]],
                                 [[1., 1., 2., 2.], [1., 1., 2., 2.]]]], requires_grad=True)
        loss = masked_feature_loss(student, teacher, torch.tensor([[[[.25, 0.]]]]))
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad[..., 0].abs().sum().item(), 0)
        self.assertEqual(student.grad[..., 1].abs().sum().item(), 0)
        student.grad = None
        empty = masked_feature_loss(student, teacher, torch.zeros((1, 1, 1, 2)))
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertEqual(student.grad.abs().sum().item(), 0)

    def test_loss_is_invariant_to_repeating_feature_channels(self):
        s = torch.randn(1, 3, 2, 2)
        t = torch.randn_like(s)
        w = torch.ones(1, 1, 2, 2)
        torch.testing.assert_close(masked_feature_loss(s, t, w),
                                   masked_feature_loss(s.repeat(1, 2, 1, 1), t.repeat(1, 2, 1, 1), w))


if __name__ == '__main__':
    unittest.main()
