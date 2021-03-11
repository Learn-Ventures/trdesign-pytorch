import os
import string
import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
from torch import nn

import config as cfg

def d(tensor=None, force_cpu = False):
    if force_cpu:
        return 'cpu'

    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    return 'cuda' if tensor.is_cuda else 'cpu'


def distance_to_bin_id(angstrom_distance):
    # Given a single (float) distance in Angstroms, return the corresponding bin_id from trRosetta['dist'] prediction
    return (np.abs(cfg.bin_dict_np['dist'] - angstrom_distance)).argmin()

# read A3M and convert letters into
# integers in the 0..max_aa range
def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(cfg.ALPHABET_full.shape[0]):
        msa[msa == cfg.ALPHABET_full[i]] = i

    # treat all unknown characters as gaps
    msa[msa > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return msa

# characters to integers
def aa2idx(seq):
    # convert letters into numbers
    abc = cfg.ALPHABET_full
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return idx

# integers back to sequence:
def idx2aa(idx):
    abc=np.array(list(cfg.ALPHABET_core_str))
    return("".join(list(abc[idx])))

def sample_distogram_bins(distogram_distribution, n_samples):
    n_residues = distogram_distribution.shape[1]
    sampled_distogram_bins = np.zeros((n_residues, n_residues)).astype(int)

    for i in range(n_residues):
        for j in range(n_residues):
            prob = distogram_distribution[:, i, j]
            prob = prob / np.sum(prob)

            samples = np.random.choice(len(distogram_distribution), n_samples, p=prob)
            sampled_distogram_bins[i,j] = int(np.mean(samples))

    return sampled_distogram_bins

def distogram_distribution_to_distogram(distribution, reduction_style = 'max', keep_no_contact_bin = False):
    '''
    This function is specific to the trRosetta distogram formatting
    '''

    distribution = np.squeeze(distribution)
    if len(distribution.shape) > 3:
        raise 'Error: distogram_distribution_to_distogram needs a single dist as input, not a batch!'

    #Remove only the special clas "no-contact":
    if keep_no_contact_bin:
        distogram_distribution = distribution
    else:
        distogram_distribution = distribution[1:]

    #Bin Distances in Angstroms:
    distances = cfg.bin_dict_np['dist'][1:]

    if keep_no_contact_bin:
        distances = np.insert(distances, 0, 22, axis=0)

    if reduction_style == 'max':
        D_pred_bins = np.argmax(distogram_distribution, axis=0)
    elif reduction_style == 'mean':
        D_pred_bins = (np.abs(distogram_distribution - np.mean(distogram_distribution, axis = 0))).argmin(axis=0)
    elif reduction_style == 'sample':
        D_pred_bins = sample_distogram_bins(distogram_distribution, 500)

    estimated_distogram = distances[D_pred_bins]
    np.fill_diagonal(estimated_distogram, 2)

    return estimated_distogram

class LinearSchedule():
    """
    This class initializes a step-varying hyperparameter (eg learning rate) that changes linearly
    To initialize it, provide two points:  [x1, y1], [x2, y2]
    It also accepts a min_value and max_value to clip potential outputs to within this range
    Finally passing in numbertype = 'int' rounds the result to the nearest integer
    """
    def __init__(self, x1, y1, x2, y2, numbertype = 'float', min_value = -np.inf, max_value = np.inf):
        super(LinearSchedule, self).__init__()
        self.slope       = (y2-y1) / (x2-x1)
        self.x1, self.y1 = x1, y1

        self.min_value   = min_value
        self.max_value   = max_value
        self.numbertype  = numbertype

    def evaluate(self, i):
        return self.slope * (i - self.x1) + self.y1

    def __call__(self, i):
        self.value = self.evaluate(i)
        self.value = np.clip(self.value, self.min_value, self.max_value)

        if self.numbertype == 'int':
            self.value = int(round(self.value))
        else:
            self.value = float(self.value)

        return self.value


######################## Plotting functions #########################

def plot_distogram(distogram, savepath, title ='', clim = None):
    plt.imshow(distogram)
    plt.title(title, fontsize=14)
    plt.xlabel('Residue i')
    plt.ylabel('Residue j')
    if clim is not None:
        plt.clim(clim[0],clim[1])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close('all')

def plot_progress(E_list, savepath, title = ''):
    x = np.array(range(len(E_list)))
    y = np.array(E_list)

    plt.plot(x, y, 'o-')
    plt.ylabel('Sequence Loss')
    plt.xlabel('N total attempted mutations')
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close('all')

def plot_gaussian(mu, sigma, savepath, title, invert):
    x_std = np.arange(mu - sigma, mu + sigma, 0.001)     # range of 1 std
    x_all = np.arange(mu - 3*sigma, mu + 3*sigma, 0.001) # entire range of x
    y_std  = norm.pdf(x_std, mu, sigma)
    y_all  = norm.pdf(x_all, mu, sigma)

    if invert:
        y_std = -1.0 * y_std
        y_all = -1.0 * y_all

    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(x_all, y_all)
    ax.fill_between(x_std, y_std, 0, alpha=0.3, color='b')
    ax.fill_between(x_all, y_all, 0, alpha=0.1)
    ax.set_xlim([mu - 3*sigma, mu + 3*sigma])
    ax.set_xlabel('Estimated metric value')
    ax.set_ylabel('Penalty')
    ax.set_title(title)
    fig.savefig(os.path.join(savepath, '%s_plot.jpg' %title), dpi=144, bbox_inches='tight')
    plt.close(fig)
