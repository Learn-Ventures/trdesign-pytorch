import numpy as np
import sys, os, time

import torch
from torch import nn
import torch.nn.functional as F

from utils import *

def to_tensor(value):
    return torch.from_numpy(np.asarray(value)).float().to(d())

def kl_div(p, q, mask = None):
    '''
    Returns the KL-div of p wrt q
    '''
    kl_div = p*torch.log(p/q)

    # Sum over the distribution_bin dimension: [batch_size, n_bins, seq_L, seq_L]
    kl_div = kl_div.sum(axis=1)

    if mask is not None:
        kl_div = kl_div * mask.unsqueeze(0)

    return kl_div.mean()

def top_prob(distogram_distributions, verbose = False):
    """
    This score can be used as a very loose proxy for a structure_models' accuracy
    Perhaps not as a loss, but more as a posterior filter for final sequences
    See Supplemental figures of trRosetta paper, Figure S3:
    https://www.pnas.org/content/pnas/suppl/2020/01/01/1914677117.DCSupplemental/pnas.1914677117.sapp.pdf
    """
    TM_score_proxies = []

    for distogram_distribution in distogram_distributions:
        distogram_distribution = distogram_distribution.cpu().detach().numpy().swapaxes(0, 2)

        # Get the LxL probabilities for the closest 12 distance bins
        w = np.sum(distogram_distribution[:,:,1:13], axis=-1)
        L = w.shape[0]

        # Extract the top diagonal matrix from the contact map probabilities:
        # TO figure out: What is this 0,8 doing here???
        idx = np.array([[i+1,j+1,0,8,w[i,j]] for i in range(L) for j in range(i+12,L)])

        # Sort these contacts according to their predicted probability:
        out = idx[np.flip(np.argsort(idx[:,4]))]

        topN=L
        if(out.shape[0]<topN):
            topN=out.shape[0]

        top = out[0:topN,4].astype(float)
        TM_score_proxy = np.mean(top)

        if verbose:
            print("Average probability of the top predicted contacts: %.2f\n" %TM_score_proxy)

        TM_score_proxies.append(TM_score_proxy)

    return TM_score_proxies


class Gaussian_Penalty(nn.Module):
    """Create a Gaussian penalty that pushes the metric towards the mean
    """
    def __init__(self, target_mu, sigma, invert_gaussian = True, savedir = None, title = ''):
        super().__init__()
        self.mu       = to_tensor(target_mu)
        self.sigma    = to_tensor(sigma)
        self.pi       = to_tensor(np.pi)
        self.var      = self.sigma ** 2
        self.normalization_constant = 1. / torch.sqrt(2*self.pi*self.var)

        #We want to sink down into the main mode, not move away from it (and we're minimizing the loss):
        self.invert = invert_gaussian

        if (savedir is not None) and (title != ''): # Dump an image of the penalty function:
            plot_gaussian(target_mu, sigma, savedir, title, self.invert)

    def compute(self, values):
        penalty = self.normalization_constant * torch.exp(-0.5 * (values - self.mu)**2 / self.var)

        if self.invert:
            penalty = -1.0 * penalty

        return penalty.mean()


class Structural_Background_Loss(nn.Module):
    """Computes the KL-divergence wrt a structural background distribution"""
    def __init__(self, seq_L, bkg_dir, weights = [1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        bkg = self.get_best_bkg_match(bkg_dir, seq_L)

        self.bd = torch.from_numpy(bkg['dist'] ).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.bo = torch.from_numpy(bkg['omega']).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.bt = torch.from_numpy(bkg['theta']).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.bp = torch.from_numpy(bkg['phi']  ).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.w  = weights

    def get_best_bkg_match(self, bkg_dir, seq_L):
        # Using some minor approximations here that's don't really affect performance
        # You can generate all the bkg distributions using this repo:
        # https://github.com/gjoni/trDesign/blob/master/01-hallucinate/src/bkgrd.py

        bkg_distribution_paths   = sorted([path for path in os.listdir(bkg_dir) if '.npz' in path])
        bkg_distribution_lengths = sorted([int(path.split('_')[-1].split('.npz')[0]) for path in bkg_distribution_paths])
        
        for L in bkg_distribution_lengths:
            if L >= seq_L:
                break

        print("Using bkg_distribution for seqL = %d as a proxy for seqL = %d" %(L, seq_L))
        bkg = dict(np.load(os.path.join(bkg_dir, 'background_distributions_%03d.npz' %L)))

        # Cut off the part we don't need:
        for key in bkg.keys():
            bkg[key] = bkg[key][:seq_L, :seq_L, :]

        return bkg

    def forward(self, pd, po, pt, pp, hallucination_mask = None):
        kl_dist       = -kl_div(pd, self.bd, mask = hallucination_mask)
        kl_omega      = -kl_div(po, self.bo, mask = hallucination_mask)
        kl_theta      = -kl_div(pt, self.bt, mask = hallucination_mask)
        kl_phi        = -kl_div(pp, self.bp, mask = hallucination_mask)

        total_background_loss = self.w[0]*kl_dist + self.w[1]*kl_omega + self.w[2]*kl_theta + self.w[3]*kl_phi
        return total_background_loss


from bisect import bisect_left
class Motif_Satisfaction(torch.nn.Module):
    """
    See https://www.biorxiv.org/content/10.1101/2020.11.29.402743v1.full.pdf,
    page 8-9, formula (2a)
    """
    def __init__(self, motif_npz_path, mask = None, save_dir = None, keys = ['dist', 'omega', 'theta', 'phi']):
        """
        motif_npz_path = t, p, d, o for the target structure
        mask           = binary mask of shape [LxL] that indicates where to apply the loss
        keys           = optionally only apply the motif_loss to certain modalities
        """
        super().__init__()

        self.device = torch.device(d())
        self.motif  = dict(np.load(motif_npz_path))
        self.mask   = mask
        self.keys   = keys
        self.seq_L  = self.mask.shape[0]

        # If the target is larger than the sequence, crop out a section of seq_L x seq_L:
        start_i = 0 #Start at the top left
        for key in self.keys:
            #self.motif[key] = self.motif[key][start_i: start_i + self.seq_L, start_i : start_i + self.seq_L]

            if save_dir is not None:
                plot_values = self.motif[key].copy()
                plot_values[plot_values == 0] = cfg.limits[key][1]
                np.fill_diagonal(plot_values, 0)
                plot_distogram(plot_values, os.path.join(save_dir, '_%s_target.jpg' %key), clim = cfg.limits[key])

        # Get the bin_indices for each of the motif_targets:
        # TODO make this allow linear combinations of the two closest bins
        self.bin_indices = {}
        for key in self.keys:
            indices = np.abs(cfg.bin_dict_np[key][np.newaxis, np.newaxis, :] - self.motif[key][:,:,np.newaxis]).argmin(axis = -1)

            # Fix the no-contact locations:
            no_contact_locations = np.argwhere(self.motif[key] == 0)
            indices[no_contact_locations[:,0], no_contact_locations[:,1]] = 0
            self.bin_indices[key] = torch.from_numpy(indices).long().to(d()).unsqueeze(0)

    def forward(self, structure_distributions):
        """ returns a loss wrt a target motif """
        pred = {}
        pred['theta'], pred['phi'], pred['dist'], pred['omega'] = structure_distributions

        # - Get the probabilities for the bins corresponding to the target motif
        # - Compute the crossentropy and average over the entire LxL matrix
        # - Multiply with the mask
        motif_loss = 0
        for key in self.keys:
            distribution = pred[key].squeeze()
            probs        = torch.gather(distribution, 0, self.bin_indices[key])
            log_probs    = torch.log(probs)
            motif_loss   -= (log_probs * self.mask).mean()

        return motif_loss
