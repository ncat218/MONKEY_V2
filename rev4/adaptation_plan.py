"""Small, dependency-free definitions for B-Clean and paired adaptation EDA."""

BCLEAN_STAGE = 'exploratory B-Clean v1'
EDA_STAGE = 'paired adaptation EDA v1'
NATIVE = (.30, .30)
COMMON = (.30, .40)
HEIGHT_CUTS = (4, 8, 16)
HEIGHT_NAMES = ('<4', '4-<8', '8-<16', '>=16')
MODEL_PAIRS = (('B0', 'B1'), ('B0', 'B-Clean'), ('B-Clean', 'B1'))
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_REPETITIONS = 2000

