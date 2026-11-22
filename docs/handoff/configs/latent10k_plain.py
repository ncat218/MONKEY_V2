# Job 88935 final checkpoint; repository root must be on PYTHONPATH.
_base_ = './b1_expanded.py'
custom_imports = dict(imports=['rev4.latent_plugins'], allow_failed_imports=False)
model = dict(type='LatentPilotDBNet', adapter_kind='plain', backbone=dict(freeze=True))
