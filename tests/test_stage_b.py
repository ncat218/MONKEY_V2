"""Synthetic checks for the Rev.4 descriptive branch."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from lowres.common import sha256
from rev4.stage_b import CONDITIONS, MANIFEST_COLUMNS, markdown, summarize, validate_generated
from submit_s2b import check_descriptive_branch


class BranchTests(TestCase):
    def test_requires_verified_lr2_and_same_s2(self):
        summary = dict(stage='S2-A', source_count=500, s2_ids_sha256='locked',
                       gates=dict(G1=True, G2=True, G3=True),
                       next_branch='LR2 selected. Blur/JPEG remains descriptive')
        check_descriptive_branch(summary, 'locked')
        with self.assertRaisesRegex(ValueError, 'different cohort'):
            check_descriptive_branch(summary, 'another')
        summary['gates']['G3'] = False
        with self.assertRaisesRegex(ValueError, 'pass G1, G2 and G3'):
            check_descriptive_branch(summary, 'locked')

    def test_all_four_variants_required_with_exact_hashes_and_canvas(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            generated = root/'variants'
            generated.mkdir()
            manifest = generated/'manifest.csv'
            image = Image.new('RGB', (21, 17), (100, 25, 10))
            records = []
            for name, (condition, parameter, fmt) in CONDITIONS.items():
                path = generated/f'{name}.{fmt.lower()}'
                image.save(path, format=fmt)
                records.append(dict(source='source.jpg', condition=condition, parameter=parameter,
                                    width=21, height=17, out_path=path.name, pillow_version='11.3.0',
                                    source_sha256='original', output_sha256=sha256(path),
                                    source_exif_orientation=1))

            def write_rows(records):
                with manifest.open('w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                    writer.writeheader()
                    writer.writerows(records)

            rows = {'source': dict(clean_size=[21, 17], clean_sha256='original')}
            write_rows(records)
            tasks, icc = validate_generated(rows, {'source.jpg': 'source'}, generated, root/'final', manifest)
            self.assertEqual(set(task['condition'] for task in tasks), set(CONDITIONS))
            self.assertEqual(icc, [])
            write_rows(records[:-1])
            with self.assertRaisesRegex(ValueError, 'cover every locked S2 source'):
                validate_generated(rows, {'source.jpg': 'source'}, generated, root/'final', manifest)
            records[0]['output_sha256'] = 'changed'
            write_rows(records)
            with self.assertRaisesRegex(ValueError, 'output SHA mismatch'):
                validate_generated(rows, {'source.jpg': 'source'}, generated, root/'final', manifest)

    def test_descriptive_metrics_do_not_assess_g3_or_change_lr2_selection(self):
        sources = ('a', 'b')
        coords = [[0, 0], [12, 0], [12, 12], [0, 12], [0, 0]]
        geom = dict(type='Polygon', coordinates=[coords])
        rows = {source: dict(roi_clean=geom, gt_clean=dict(valid=[dict(ann_id='word', geometry=geom)],
                                                           ignored=[])) for source in sources}

        def sample(hit):
            predictions = dict(polygons=[[0, 0, 12, 0, 12, 12, 0, 12]] if hit else [],
                               scores=[.8] if hit else [])
            return dict(metrics=dict(counts=[int(hit), 1, int(hit), int(hit)], matched_ann_ids=['word'] if hit else []),
                        prediction=predictions, candidates_before_score=predictions, latency_ms=1.0,
                        gt={'word': dict(native_height=12.0, network_input_height=36.0,
                                         kernel_pixels=12, coverage=1.0 if hit else .2)})

        results = {condition: {source: sample(condition in ('clean', 'jpeg30')) for source in sources}
                   for condition in ('clean', *CONDITIONS)}
        report = summarize(results, rows, dict(match_iou=.5, word_coverage=.5))
        self.assertEqual(report['selected_condition'], 'lr2')
        self.assertEqual(report['bootstrap_repetitions'], 2000)
        self.assertEqual(report['conditions']['blur1']['gates'], dict(G1=True, G2=True, G3='NOT ASSESSED'))
        self.assertEqual(report['conditions']['jpeg30']['gates'], dict(G1=False, G2=False, G3='NOT ASSESSED'))
        self.assertEqual(report['conditions']['blur1']['primary_failure_counts'], {'weak_response': 2})
        self.assertEqual(report['tau'], 'N/A')
        report['baseline_job'] = 'synthetic'
        self.assertIn('NOT ASSESSED', markdown(report))
