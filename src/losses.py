"""Components of the loss function."""

# native
from pathlib import Path

# lib
import numpy as np
import torch

# pkg
from utils import d, plot_distogram
import config as cfg


def to_tensor(value):
    """Return a pytorch tensor from a python list."""
    return torch.from_numpy(np.asarray(value)).float().to(d())


def kl_div(p, q, mask=None):
    """Return the KL-divergence of `p` with respect to `q`."""
    kl_divergence = p * torch.log(p / q)

    # Sum over the distribution_bin dimension: [batch_size, n_bins, seq_L, seq_L]
    kl_divergence = kl_divergence.sum(axis=1)

    if mask is not None:
        kl_divergence *= mask.unsqueeze(0)

    return kl_divergence.mean()


def top_prob(distogram_distributions, verbose=False):
    """Return a list of TM score proxies.

    This score can be used as a very loose proxy for the accuracy of a structure model.
    Perhaps not as a loss, but more as a posterior filter for final sequences.

    For more information, see Figure S3 in the [trRosetta Supplemental Figures][1].

    [1]: https://www.pnas.org/content/pnas/suppl/2020/01/01/1914677117.DCSupplemental/pnas.1914677117.sapp.pdf#page=4
    """
    TM_score_proxies = []

    for distogram_distribution in distogram_distributions:
        distogram_distribution = (
            distogram_distribution.cpu().detach().numpy().swapaxes(0, 2)
        )

        # Get the LxL probabilities for the closest 12 distance bins
        w = np.sum(distogram_distribution[:, :, 1:13], axis=-1)
        L = w.shape[0]

        # Extract the top diagonal matrix from the contact map probabilities:
        # TODO: figure out what the 0,8 doing here?
        idx = np.array(
            [[i + 1, j + 1, 0, 8, w[i, j]] for i in range(L) for j in range(i + 12, L)]
        )

        # Sort these contacts according to their predicted probability:
        out = idx[np.flip(np.argsort(idx[:, 4]))]

        topN = L
        if out.shape[0] < topN:
            topN = out.shape[0]

        top = out[0:topN, 4].astype(float)
        TM_score_proxy = np.mean(top)
        TM_score_proxies.append(TM_score_proxy)

        if verbose:
            print(f"Avg probability of top predicted contacts: {TM_score_proxy:.2f}\n")

    return TM_score_proxies


class Structural_Background_Loss(torch.nn.Module):
    """Computes the KL-divergence wrt a structural background distribution."""

    DEFAULT_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

    def __init__(self, seq_L, bkg_dir, weights=None):
        """Construct a loss object."""
        super().__init__()
        bkg = self.get_best_bkg_match(bkg_dir, seq_L)

        self.bd = torch.from_numpy(bkg["dist"]).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.bo = torch.from_numpy(bkg["omega"]).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.bt = torch.from_numpy(bkg["theta"]).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.bp = torch.from_numpy(bkg["phi"]).permute(2, 1, 0).to(d()).unsqueeze(0)
        self.w = weights or Structural_Background_Loss.DEFAULT_WEIGHTS

    @staticmethod
    def get_best_bkg_match(bkg_dir, seq_L):
        """Return the best background distribution for the given sequence length.

        This method uses some minor approximations that's don't really affect performance.
        For additional lengths, you can generate background distributions using:
        https://github.com/gjoni/trDesign/blob/master/01-hallucinate/src/bkgrd.py
        """
        bkg_path, L = None, 0
        for bkg_path in sorted(Path(bkg_dir).glob("*.npz")):
            L = int(bkg_path.stem.split("_")[-1])
            if L >= seq_L:
                break

        print(f"Using {bkg_path.name} as a proxy for seq_L = {seq_L}")
        bkg = dict(np.load(bkg_path))

        # cut off the part we don't need
        for key in bkg:
            bkg[key] = bkg[key][:seq_L, :seq_L, :]

        return bkg

    def forward(self, pd, po, pt, pp, hallucination_mask=None):
        """Return the total background loss."""
        kl_dist = -kl_div(pd, self.bd, mask=hallucination_mask)
        kl_omega = -kl_div(po, self.bo, mask=hallucination_mask)
        kl_theta = -kl_div(pt, self.bt, mask=hallucination_mask)
        kl_phi = -kl_div(pp, self.bp, mask=hallucination_mask)

        # TODO: consider np.dot?
        parts = [kl_dist, kl_omega, kl_theta, kl_phi]
        total_background_loss = sum([self.w[i] * p for i, p in enumerate(parts)])

        return total_background_loss


class Motif_Satisfaction(torch.nn.Module):
    """Compute a loss compared to motif specified by an `.npz` file.

    For more information, see formula 2a in [Tisher et al. (2020)][1].
    [1]: https://www.biorxiv.org/content/10.1101/2020.11.29.402743v1.full.pdf#page=8
    """

    DEFAULT_KEYS = ["dist", "omega", "theta", "phi"]

    def __init__(self, motif_npz_path, mask=None, save_dir=None, keys=None):
        """Construct a loss object.

        Args:
            motif_npz_path: t, p, d, o for the target structure
            mask: binary mask of shape [LxL] that indicates where to apply the loss
            keys (optional): only apply the `motif_loss` to certain components
                (default: ["dist", "omega", "theta", "phi"])
        """
        super().__init__()

        self.device = torch.device(d())
        self.motif = dict(np.load(motif_npz_path))
        self.mask = mask
        self.keys = keys or Motif_Satisfaction.DEFAULT_KEYS
        self.seq_L = self.mask.shape[0]

        save_dir = Path(save_dir) if save_dir else None

        # If the target is larger than the sequence, crop out a section of seq_L x seq_L:
        # start_i = 0  # Start at the top left
        for key in self.keys:
            # self.motif[key] = self.motif[key][start_i: start_i + self.seq_L, start_i : start_i + self.seq_L]

            if save_dir:
                plot_values = self.motif[key].copy()
                plot_values[plot_values == 0] = cfg.limits[key][1]
                np.fill_diagonal(plot_values, 0)
                plot_distogram(
                    plot_values,
                    save_dir / f"_{key}_target.jpg",
                    clim=cfg.limits[key],
                )

        # Get the bin_indices for each of the motif_targets:
        # TODO make this allow linear combinations of the two closest bins
        self.bin_indices = {}
        for key in self.keys:
            indices = np.abs(
                cfg.bin_dict_np[key][np.newaxis, np.newaxis, :]
                - self.motif[key][:, :, np.newaxis]
            ).argmin(axis=-1)

            # Fix the no-contact locations:
            no_contact_locations = np.argwhere(self.motif[key] == 0)
            indices[no_contact_locations[:, 0], no_contact_locations[:, 1]] = 0
            self.bin_indices[key] = (
                torch.from_numpy(indices).long().to(d()).unsqueeze(0)
            )

    def forward(self, structure_distributions):
        """ returns a loss wrt a target motif """
        pred = {}
        (
            pred["theta"],
            pred["phi"],
            pred["dist"],
            pred["omega"],
        ) = structure_distributions

        # - Get the probabilities for the bins corresponding to the target motif
        # - Compute the crossentropy and average over the entire LxL matrix
        # - Multiply with the mask
        motif_loss = 0
        for key in self.keys:
            distribution = pred[key].squeeze()
            probs = torch.gather(distribution, 0, self.bin_indices[key])
            log_probs = torch.log(probs)
            motif_loss -= (log_probs * self.mask).mean()

        return motif_loss
