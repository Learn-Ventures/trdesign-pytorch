#!/usr/bin/env python
"""Run the trDesign loop."""

# native
from pathlib import Path
import linecache
import sys

# lib
import numpy as np

# pkg
# pylint: disable=wrong-import-position
script_dir = Path(__file__).parent
sys.path[0:0] = [str(script_dir / "src"), str(script_dir)]

import mcmc
import utils
import config as cfg

def get_sequence(i, L, aa_valid, seed_file=None):
    """Return a sequence of length `L`.

    If `seed_file` is provided, return the first `L` characters of line `i`.
    Otherwise, return a completely random sequence using `aa_valid` symbols.
    """
    return (
        linecache.getline(seed_file, i + 1)[:L]
        if seed_file
        else utils.idx2aa(np.random.choice(aa_valid, L))
    )


def main():
    """Run the trDesign loop."""

    ########################################################
    # get valid residues
    ########################################################

    # any residue types to skip during sampling?
    aa_valid = np.arange(20)
    if cfg.RM_AA:
        aa_skip = utils.aa2idx(cfg.RM_AA.replace(",", ""))
        aa_valid = np.setdiff1d(aa_valid, aa_skip)

    ########################################################
    # run MCMC
    ########################################################

    seqs, seq_metrics = [], []
    for i in range(cfg.num_simulations):
        print("#####################################")
        print(f"\n --- Optimizing sequence {i:04} of {cfg.num_simulations:04}...")

        mcmc_optim = mcmc.MCMC_Optimizer(
            cfg.LEN,
            cfg.AA_WEIGHT,
            cfg.MCMC,
            cfg.native_freq,
            cfg.experiment_name,
            aa_valid,
            max_aa_index=cfg.MAX_AA_INDEX,
            sequence_constraint=cfg.sequence_constraint,
            target_motif_path=cfg.target_motif_path,
        )

        start_seq = get_sequence(i, cfg.LEN, aa_valid, seed_file=cfg.seed_filepath)

        metrics = mcmc_optim.run(start_seq)
        seqs.append(metrics["sequence"])
        seq_metrics.append(metrics)


if __name__ == "__main__":
    main()
