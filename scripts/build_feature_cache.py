"""Build the cached deal features used by `run_experiments.py exploit`."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fcd.game import Config
import run_experiments as R
for ranks, abst in ((6, "base"), (6, "fine"), (5, "base"), (7, "base")):
    cfg = Config(n_ranks=ranks, abstraction=abst)
    for n, seed in ((1_000_000, R.FIT_SEED), (300_000, R.EVAL_SEED)):
        R.features(cfg, n, seed)
        print("cached", cfg.label(), abst, n, flush=True)
