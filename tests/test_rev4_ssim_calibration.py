"""Synthetic unit tests only. No dataset files or real reading responses are used."""
from collections import Counter
from unittest import TestCase

from PIL import Image

from rev4.ssim_calibration import (
    BIN_STARTS, QUOTAS, bin_index, blind_layout, canonical_ids, crop_ssim, match_two_readers,
    pick_pool, population, require_lr2_failure, select_cards, weighted_curve, word_crops,
)


def ids(count):
    return [('source', f'word-{i:05d}') for i in range(count)]


def box(x0, y0, x1, y1):
    return dict(type='Polygon', coordinates=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]])


class PoolTests(TestCase):
    def test_conditional_branch_requires_confirmed_failure(self):
        with self.assertRaises(ValueError):
            require_lr2_failure(dict(G1=True, G2=True))
        with self.assertRaises(ValueError):
            require_lr2_failure(dict(G1=True, G2=True, G3=True))
        require_lr2_failure(dict(G1=True, G2=True, G3=False))

    def test_fixed_pool_is_unique_canonical_and_order_independent(self):
        words = ids(20_007)
        a = pick_pool(words)
        self.assertEqual(len(a), 20_000)
        self.assertEqual(a, pick_pool(list(reversed(words))+words[:7]))
        self.assertEqual(pick_pool(words[:3]), words[:3])
        self.assertEqual(canonical_ids(words[:2]), b'source\tword-00000\nsource\tword-00001\n')

    def test_bin_boundaries_and_invalid_counts(self):
        self.assertEqual([bin_index(x) for x in BIN_STARTS], list(range(7)))
        self.assertEqual(bin_index(1.00), 6)
        values = [None, -.01, .29, .30, .90, 1.00, 1.001]
        bins, excluded = population(dict(zip(ids(len(values)), values)))
        self.assertEqual([len(group) for group in bins], [1, 0, 0, 0, 0, 0, 2])
        self.assertEqual(excluded, dict(invalid=1, below_0_30=2, below_zero=1, above_one=1))

    def test_quotas_and_disjoint_conditions(self):
        scores = {word: BIN_STARTS[i//180] for i, word in enumerate(ids(7*180))}
        combined = {'Blur1': scores, 'JPEG30': scores}
        locked = select_cards(combined)
        self.assertEqual(locked, select_cards(combined))
        for condition in combined:
            selected = locked[condition]['selected']
            self.assertEqual(len(selected), 200)
            self.assertEqual(Counter(bin_index(scores[word]) for word in selected),
                             Counter(dict(enumerate(QUOTAS))))
        self.assertEqual(locked['JPEG30']['overlap'], 0)
        with self.assertRaises(ValueError):
            select_cards({'Blur1': scores, 'JPEG30': {}})

    def test_shortfall_prefers_highest_ssim_bin(self):
        words = ids(300)
        scores = dict.fromkeys(words[:20], .31) | dict.fromkeys(words[20:120], .81) | dict.fromkeys(words[120:], .95)
        locked = select_cards({'Blur1': scores, 'JPEG30': scores})['Blur1']
        self.assertEqual(Counter(bin_index(scores[word]) for word in locked['selected']), {0: 20, 5: 20, 6: 160})


class CropAndReadingTests(TestCase):
    def test_crops_clamp_and_keep_small_crops_invalid(self):
        clean = Image.new('RGB', (14, 14), (70, 110, 90))
        self.assertEqual(word_crops(clean, clean, box(-.1, -.1, 8.1, 8.1))[0].size, (9, 9))
        self.assertEqual(crop_ssim(clean, clean, box(0, 0, 8, 8)), 1.0)
        self.assertIsNone(crop_ssim(clean, clean, box(0, 0, 6, 8)))
        with self.assertRaises(ValueError):
            word_crops(clean, Image.new('RGB', (15, 14)), box(0, 0, 8, 8))

    def test_public_cards_hide_truth_and_require_two_complete_readers(self):
        word, = ids(1)
        selections = {'Blur1': {'selected': [word]}, 'JPEG30': {'selected': []}}
        scores = {'Blur1': {word: .95}, 'JPEG30': {}}
        public, private = blind_layout(selections, scores, {word: 'Á.'}, seed=17)
        self.assertTrue(all(set(item) == {'card_id', 'image'} for item in public))
        a = {card: ' A\u0301. ' for card in private}
        b = dict(a)
        bad = next(card for card, item in private.items() if item['version'] == 'degraded')
        b[bad] = 'Á'
        outcomes, agreement = match_two_readers(private, a, b)
        self.assertEqual(outcomes['Blur1'][word], {'clean': True, 'degraded': False})
        self.assertEqual(agreement['Blur1']['degraded'], {'agree': 0, 'total': 1})
        with self.assertRaises(ValueError):
            match_two_readers(private, a, {})


class ScoringTests(TestCase):
    def test_weighted_rate_differs_from_stratified_sample_rate(self):
        words = ids(100)
        scores = dict.fromkeys(words[:90], .55) | dict.fromkeys(words[90:], .95)
        selected = words[:10]+words[90:]
        results = {word: dict(clean=True, degraded=word in words[90:]) for word in selected}
        summary = weighted_curve(scores, selected, results)
        self.assertAlmostEqual(summary['thresholds'][0]['R_weighted'], .10)
        self.assertEqual(summary['thresholds'][0]['R_unweighted'], .50)
        self.assertEqual(summary['thresholds'][-1]['status'], 'UNDERPOWERED')
        self.assertEqual(summary['bins'][0]['status'], 'not observed in pool')
        self.assertIsNone(summary['tau'])
        with self.assertRaises(ValueError):
            weighted_curve(scores, words[90:], {word: results[word] for word in words[90:]})

    def test_rejects_tau_when_higher_grid_is_underpowered(self):
        words = ids(60)
        scores = dict.fromkeys(words[:40], .85) | dict.fromkeys(words[40:], .95)
        outcomes = {word: dict(clean=True, degraded=True) for word in words}
        result = weighted_curve(scores, words, outcomes)
        self.assertEqual(result['thresholds'][0]['status'], 'PASS')
        self.assertEqual(result['thresholds'][-1]['N_eff'], 20.0)
        self.assertIsNone(result['tau'])

    def test_synthetic_cutoff_satisfies_entire_upper_grid(self):
        words = ids(80)
        scores = dict.fromkeys(words[:40], .60) | dict.fromkeys(words[40:], .95)
        outcomes = {word: dict(clean=True, degraded=word in words[40:]) for word in words}
        result = weighted_curve(scores, words, outcomes)
        self.assertEqual(result['tau'], .65)
        self.assertEqual(result['thresholds'][-1]['N_eff'], 40.0)
