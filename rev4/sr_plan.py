"""Fixed B1 weights and operating points; pretrained SR inference only."""
BRANCHES = ('native', 'bicubic_x2', 'edsr_x2')
PAIRS = ((.3, .4), (.4, .6))
PRIMARY = 1
RECIPE = dict(detector='completed B1 job12187, selected 10000-update checkpoint; frozen inference',
    restoration='same pinned torchSR EDSR64 x2; no training', branches=list(BRANCHES),
    operating_points=[list(v) for v in PAIRS], primary_pair=list(PAIRS[PRIMARY]), threshold_selection='none',
    geometry=dict(text_repr_type='quad', unclip_ratio=1.5, min_text_width=5, epsilon_ratio=.01, max_candidates=3000),
    sources='2262 remaining-validation; same fixed 500-source stress cohort as fusion111683',
    conditions=['clean','lr2','lr4','lr8','blur1','blur2','jpeg30','jpeg10'],
    deployment='always-on x2 for every input including Clean; no ground-truth/LR-oracle router',
    height_bins='native LR2 reference of original source, fixed across restoration/conditions',
    precision='fp32; matmul TF32 off, cuDNN TF32 on; same frozen runtime',
    tile=256, halo=40, bootstrap_repetitions=2000, bootstrap_seed=20261107,
    timing='SR/read + detector/quad decode at primary .40/.60; excludes PNG encoding, hashes, metric evaluation; non-isolated',
    no_test=True)
