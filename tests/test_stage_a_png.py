"""Synthetic regression for PNG ICC metadata. It opens no S2 data."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image, ImageCms, PngImagePlugin
from rev4.stage_a import PILLOW_CHUNK_LIMIT_ERROR, open_native_lr, verify_detector_png_decode


class NativePngTests(TestCase):
    def test_oversized_icc_is_omitted_only_in_temporary_view(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'oversized_icc.png'
            rgb = Image.new('RGB', (3, 2), (18, 93, 217))
            rgb.save(path, format='PNG', icc_profile=b'a'*(PngImagePlugin.MAX_TEXT_CHUNK+65536))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, PILLOW_CHUNK_LIMIT_ERROR):
                Image.open(path)
            image, original_with_icc = open_native_lr(path)
            with image:
                self.assertEqual(image.size, rgb.size)
                self.assertEqual(image.convert('RGB').tobytes(), rgb.tobytes())
                self.assertNotIn('icc_profile', image.info)
            self.assertEqual(original_with_icc, original)
            self.assertEqual(path.read_bytes(), original)

    def test_ordinary_png_keeps_standard_pillow_decode(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'ordinary.png'
            Image.new('RGB', (2, 2)).save(path, format='PNG')
            image, original_with_icc = open_native_lr(path)
            with image:
                self.assertEqual(image.size, (2, 2))
            self.assertIsNone(original_with_icc)

    def test_frozen_opencv_reader_matches_rgb_with_large_valid_icc(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'large_srgb.png'
            profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
            length = PngImagePlugin.MAX_TEXT_CHUNK+65536
            profile.extend(b'\0'*(length-len(profile)))
            profile[:4] = length.to_bytes(4, 'big')
            rgb = Image.new('RGB', (4, 3), (18, 93, 217))
            rgb.save(path, format='PNG', icc_profile=bytes(profile))
            image, raw = open_native_lr(path)
            with image:
                self.assertIsNotNone(raw)
                verify_detector_png_decode(raw, image, path)
                self.assertEqual(image.convert('RGB').tobytes(), rgb.tobytes())
