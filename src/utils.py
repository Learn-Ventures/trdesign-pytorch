"""Helpful utilities.

Many of these are based on the work by @lucidrains:
https://github.com/lucidrains/tr-rosetta-pytorch/blob/main/tr_rosetta_pytorch/utils.py
"""

# native
from pathlib import Path
import string

# lib
from matplotlib import pylab as plt
import numpy as np
import torch

# pkg
import config as cfg


def d(tensor=None, force_cpu=False):
    """Return 'cpu' or 'cuda' depending on context."""
    if force_cpu:
        return "cpu"
    if tensor is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if tensor.is_cuda else "cpu"


def distance_to_bin_id(angstrom_distance: float):
    """Return the `bin_id` for a distance from a structure prediction."""
    # Given a single (float) distance in Angstroms,
    # return the corresponding bin_id from trRosetta['dist'] prediction
    return (np.abs(cfg.bin_dict_np["dist"] - angstrom_distance)).argmin()

def average_dict(list_of_dicts, detach = False):
    """Returns a dict where each entry contains the average of the tensors for that key"""
    averaged_outputs = {}
    for key in list_of_dicts[0].keys():
        key_values = []
        for dict_el in list_of_dicts:
            key_values.append(dict_el[key])

        averaged_outputs[key] = torch.stack(key_values).mean(axis=0)

        if detach:
            averaged_outputs[key] = averaged_outputs[key].cpu().detach().numpy()

    return averaged_outputs

def parse_a3m(filename):
    """Return the contents of an `.a3m` file as integers.

    The resulting integers are in the range 0..max_aa.
    """
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in Path(filename).open():
        # skip labels
        if line[0] != ">":
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i in range(cfg.ALPHABET_full.shape[0]):
        msa[msa == cfg.ALPHABET_full[i]] = i

    # treat all unknown characters as gaps
    msa[msa > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return msa


def aa2idx(seq: str) -> np.ndarray:
    """Return the sequence of characters as a list of integers."""
    # convert letters into numbers
    abc = cfg.ALPHABET_full
    idx = np.array(list(seq), dtype="|S1").view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return idx


def idx2aa(idx: np.ndarray) -> str:
    """Return the string representation from an array of integers."""
    abc = np.array(list(cfg.ALPHABET_core_str))
    return "".join(list(abc[idx]))


def sample_distogram_bins(distogram_distribution, n_samples):
    """Return a distogram sampled from a distogram distribution."""
    n_residues = distogram_distribution.shape[1]
    sampled_distogram_bins = np.zeros((n_residues, n_residues)).astype(int)

    for i in range(n_residues):
        for j in range(n_residues):
            prob = distogram_distribution[:, i, j]
            prob = prob / np.sum(prob)

            samples = np.random.choice(len(distogram_distribution), n_samples, p=prob)
            sampled_distogram_bins[i, j] = int(np.mean(samples))

    return sampled_distogram_bins


def distogram_distribution_to_distogram(
    distribution, reduction_style="max", keep_no_contact_bin=False
):
    """Return the estimated distogram from a distribution of distograms.

    NOTE: This function is specific to the trRosetta distogram format.
    """

    distribution = np.squeeze(distribution)
    if len(distribution.shape) > 3:
        raise "Error: distogram_distribution_to_distogram needs a single dist as input, not a batch!"

    # Remove only the special class "no-contact":
    if keep_no_contact_bin:
        distogram_distribution = distribution
    else:
        distogram_distribution = distribution[1:]

    # Bin Distances in Angstroms:
    distances = cfg.bin_dict_np["dist"][1:]

    if keep_no_contact_bin:
        distances = np.insert(distances, 0, 22, axis=0)

    if reduction_style == "max":
        D_pred_bins = np.argmax(distogram_distribution, axis=0)
    elif reduction_style == "mean":
        D_pred_bins = (
            np.abs(distogram_distribution - np.mean(distogram_distribution, axis=0))
        ).argmin(axis=0)
    elif reduction_style == "sample":
        D_pred_bins = sample_distogram_bins(distogram_distribution, 500)

    estimated_distogram = distances[D_pred_bins]
    np.fill_diagonal(estimated_distogram, 2)

    return estimated_distogram


### Plotting Function ###


def plot_distogram(distogram, savepath, title="", clim=None):
    """Save a plot of a distogram to the given path."""
    plt.imshow(distogram)
    plt.title(title, fontsize=14)
    plt.xlabel("Residue i")
    plt.ylabel("Residue j")
    if clim is not None:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close("all")


def plot_progress(E_list, savepath, title=""):
    """Save a plot of sequence losses to the given path."""
    x = np.array(range(len(E_list)))
    y = np.array(E_list)

    plt.plot(x, y, "o-")
    plt.ylabel("Sequence Loss")
    plt.xlabel("N total attempted mutations")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close("all")
