# Portable inference override; use only model/test_pipeline, never start a Runner.
_base_ = '../inference_config.py'
textdet_parameter_precision = 'fp32'
model = dict(det_head=dict(postprocessor=dict(
    text_repr_type='quad', mask_thr=0.3, min_text_score=0.4,
    min_text_width=5, unclip_ratio=1.5, epsilon_ratio=0.01, max_candidates=3000)))
