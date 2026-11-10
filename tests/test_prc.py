"""Regression checks for the PRC geometry and loss boundaries."""
import importlib.util
from unittest import TestCase, skipUnless

import numpy as np

from rev4.prc_geometry import clean_canvas, masked_response_loss


class PRCTests(TestCase):
    def test_flip_and_crop_keep_pixel_centers(self):
        clean = np.arange(24, dtype=np.uint8).reshape(4, 6)
        flip = np.array([[-1., 0., 6.], [0., 1., 0.]])
        actual, valid = clean_canvas(clean, (4, 6), flip, (1, 1), (1., 1.), (2, 3))
        np.testing.assert_array_equal(actual, clean[:, ::-1][1:3, 1:4])
        self.assertTrue(valid.all())

    def test_odd_clean_size_and_outside_canvas(self):
        import cv2

        clean = np.arange(35, dtype=np.uint8).reshape(5, 7)
        translated = np.array([[1., 0., 1.], [0., 1., 1.]])
        actual, valid = clean_canvas(clean, (2, 3), translated, (0, 0), (1., 1.), (4, 5))
        expected = np.zeros((4, 5), dtype=bool)
        expected[1:3, 1:4] = True
        np.testing.assert_array_equal(valid, expected)
        np.testing.assert_allclose(actual[valid], cv2.resize(clean, (3, 2)).ravel(), atol=1)
        self.assertFalse(actual[~valid].any())

    @skipUnless(importlib.util.find_spec('torch'), 'Requires torch')
    def test_only_matched_mask_pixels_receive_gradients(self):
        import torch

        student = torch.tensor([.25, .25, .25], requires_grad=True)
        teacher = torch.tensor([.75, .75, .75], requires_grad=True)
        loss = masked_response_loss(student, teacher, torch.tensor([True, False, True]))
        self.assertEqual(loss.item(), .25)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([-.5, 0., -.5]))
        self.assertIsNone(teacher.grad)
        student.grad = None
        loss = masked_response_loss(student, teacher, torch.zeros(3, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros(3))
