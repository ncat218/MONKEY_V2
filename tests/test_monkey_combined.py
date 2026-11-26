import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lowres.common import read_json,sha256
from rev4.durable_io import save_json,write_text
import submit_monkey_combined as submit


class CombinedTests(unittest.TestCase):
    def test_one_allocation_keeps_original_study_identities_and_outputs(self):
        before=submit.children()
        identities={k:copy.deepcopy(v.launch_identity) for k,v in before.items()}
        a=submit.light();after=submit.children()
        self.assertEqual(a.launch_identity['children'],identities)
        self.assertEqual({k:v.launch_identity for k,v in after.items()},identities)
        self.assertEqual(before['quality'].output.parent.name,'tri_quality_combo')
        self.assertEqual(before['research'].output.parent.name,'research_bundle')
        self.assertEqual(a.output.parent.name,'monkey_combined')

    def test_held_replacement_has_one_gpu_and_parent_dependency(self):
        a=submit.light();cmd=submit.command(a,'163064',True)
        self.assertEqual(cmd.count('--gres=gpu:2g.35gb:1'),1)
        self.assertIn('--hold',cmd)
        self.assertIn('--dependency=afterok:163064',cmd)
        self.assertIn('--time=2-06:00:00',cmd)
        self.assertNotIn('scancel',' '.join(cmd))
        self.assertNotIn('--hold',submit.command(a,None,False))

    def test_rejects_changed_child_launch_before_any_execution(self):
        with TemporaryDirectory() as d:
            roots={n:SimpleNamespace(output=Path(d)/n,launch_identity={'source':{'file':'sha'}}) for n in submit.STUDIES}
            for ctx in roots.values():save_json(ctx.output/'submission/launch.json',ctx.launch_identity)
            a=SimpleNamespace(launch_identity={'children':{n:ctx.launch_identity for n,ctx in roots.items()}})
            with patch.object(submit,'children',return_value=roots):
                submit.verify_child_launches(a)
                save_json(roots['research'].output/'submission/launch.json',{'changed':True})
                with self.assertRaisesRegex(ValueError,'original queued launch differs'):
                    submit.verify_child_launches(a)

    def test_child_failure_preserves_progress_and_stops_next_study(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            a=SimpleNamespace(output=root,python='python',launch_identity={'test':True})
            snapshots={n:SimpleNamespace(output=root/n) for n in submit.STUDIES}
            with patch.object(submit.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as run:
                with self.assertRaisesRegex(ValueError,'next study will not run'):
                    submit.run_studies(a,snapshots)
            self.assertEqual(run.call_count,1)
            self.assertEqual(read_json(root/'status.json')['quality']['state'],'FAIL')
            self.assertNotIn('research',read_json(root/'status.json'))
            self.assertFalse((root/'results/complete.json').exists())

    def test_serial_execution_writes_one_index_and_resumes_passed_child(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            a=SimpleNamespace(output=root,python='python',launch_identity={'test':True})
            snapshots={n:SimpleNamespace(output=root/n) for n in submit.STUDIES}
            verified=lambda n,ctx:dict(report=n+'.md',report_sha256='sha',completion_sha256='sha')
            with patch.object(submit.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run, \
                 patch.object(submit,'completion',side_effect=verified):
                submit.run_studies(a,snapshots)
                self.assertIn('submit_quality_combo.py',run.call_args_list[0].args[0][2])
                self.assertIn('submit_research_bundle.py',run.call_args_list[1].args[0][2])
                self.assertEqual(run.call_count,2)
                submit.run_studies(a,snapshots)
                self.assertEqual(run.call_count,2)
            self.assertEqual(set(read_json(root/'results/complete.json')['children']),set(submit.STUDIES))

    def test_exit_zero_without_receipt_is_not_success(self):
        with TemporaryDirectory() as d:
            ctx=SimpleNamespace(output=Path(d))
            with self.assertRaisesRegex(ValueError,'without a completion receipt'):
                submit.completion('quality',ctx)

    def test_partial_bundle_is_not_reported_as_complete(self):
        with TemporaryDirectory() as d:
            root=Path(d);ctx=SimpleNamespace(output=root)
            write_text(root/'results/bundle_report.md','Partial results\n')
            save_json(root/'identity.json',{'source':'unchanged'})
            marker=dict(identity={'source':'unchanged'},no_test=True,final=False,
                report_sha256=sha256(root/'results/bundle_report.md'))
            save_json(root/'results/complete.json',marker)
            with self.assertRaisesRegex(ValueError,'incomplete/failed phases'):
                submit.completion('research',ctx)
            marker['final']=True;save_json(root/'results/complete.json',marker)
            self.assertEqual(submit.completion('research',ctx)['report_sha256'],marker['report_sha256'])


if __name__=='__main__':unittest.main()
