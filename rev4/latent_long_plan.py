"""Registered long-budget extension; does not replace the 1000-update pilot."""
from copy import deepcopy
from rev4.latent_plan import ARMS,PAIRS,RECIPE as PILOT

RECIPE=deepcopy(PILOT)
RECIPE.update(optimizer_updates=10000,warmup_updates=100,checkpoint_every_updates=2500,
              max_keep_checkpoints=4,checkpoint_selection='final 10000 updates only',
              scheduler='Linear .1 to 1 for 100 optimizer updates, then Poly .9 to LR*.01',
              initialization='fresh B1 weights and fresh optimizer; zero adapter output; no pilot resume')
