import warnings, os, sys, shutil

warnings.filterwarnings('ignore',category=FutureWarning)
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, script_dir+'/src/')

import numpy as np
from utils import *
from mcmc import *
import config as cfg

def print_section(pre_newlines = 0, post_newlines = 0, num_spacers = 40):
    [print("\n") for i in range(pre_newlines)]
    [print("#"*num_spacers + "\n" + "#"*num_spacers)]
    [print("\n") for i in range(post_newlines)]

def get_sequence(i, L, aa_valid, seed_file = None):
    if seed_file is None:
        return idx2aa(np.random.choice(aa_valid, L))
    else:
        fp = open(seed_file)
        for line_i, line in enumerate(fp):
            if line_i == i:
                return line[:L]

def main():

    ########################################################
    # get valid residues
    ########################################################

    # any residue types to skip during sampling?
    aa_valid = np.arange(20)
    if cfg.RM_AA != "":
        aa_skip = aa2idx(cfg.RM_AA.replace(',',''))
        aa_valid = np.setdiff1d(aa_valid, aa_skip)

    ########################################################
    # run MCMC
    ########################################################

    seqs, seq_metrics = [], []

    for i in range(cfg.num_simulations):
        print_section(pre_newlines = 2, post_newlines = 2)
        print("Optimizing sequence %04d of %04d..." %(i, cfg.num_simulations))

        mcmc_optim    = MCMC_Optimizer(cfg.LEN, cfg.AA_WEIGHT, cfg.MCMC, cfg.native_freq, cfg.experiment_name, 
            aa_valid,
            max_aa_index = cfg.MAX_AA_INDEX,
            sequence_constraint = cfg.sequence_constraint,
            target_motif_path = cfg.target_motif_path)
        
        start_seq     = get_sequence(i, cfg.LEN, aa_valid, seed_file = cfg.seed_filepath)

        metrics  = mcmc_optim.run(start_seq)
        seqs.append(metrics['sequence'])
        seq_metrics.append(metrics)


if __name__ == '__main__':
    main()