"""Declared exploratory grid. No model training or TEST phase exists here."""
STAGE = 'PR quad and FP diagnosis v1'
SCORES = (.20, .30, .35, .40, .45, .50, .60, .70)
GEOMETRIES = (
    ('base', .30, 1.5, 5, .010),
    ('unclip_1p2', .30, 1.2, 5, .010),
    ('unclip_1p8', .30, 1.8, 5, .010),
    ('width_3', .30, 1.5, 3, .010),
    ('width_7', .30, 1.5, 7, .010),
    ('epsilon_005', .30, 1.5, 5, .005),
    ('epsilon_020', .30, 1.5, 5, .020),
    ('mask_020', .20, 1.5, 5, .010),
    ('mask_040', .40, 1.5, 5, .010),
)
INSTALLED_PP_SHA256 = '9bfff8aaaca103dd08d7565f2b7a5410ab20a6427fe1f0baad5a5d04d207c8b8'
REVIEW_SEED = 20261119
BOOTSTRAP_SEED = 20261107
REPETITIONS = 2000


def grid():
    return [dict(geometry=name, mask_thr=mask, unclip_ratio=unclip,
                 min_text_width=width, epsilon_ratio=epsilon, min_text_score=score)
            for name, mask, unclip, width, epsilon in GEOMETRIES for score in SCORES]


def setting_key(setting):
    return f'{setting["geometry"]}/score_{setting["min_text_score"]:.2f}'


NATIVE_INDEX = 1
COMMON_INDEX = 3
