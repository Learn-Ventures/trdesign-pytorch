"""Markov Chain Monte Carlo for trDesign."""

# native
from datetime import datetime
import time

# lib
import torch
import numpy as np

# pkg
from losses import *  # pylint: disable=wildcard-import, unused-wildcard-import
from tr_Rosetta_model import trRosettaEnsemble, prep_seq
from utils import aa2idx, distogram_distribution_to_distogram, idx2aa, plot_progress
import config as cfg


def v(torch_value):
    """Return a detached value, if possible."""
    try:
        return torch_value.cpu().detach().item()
    except Exception:
        return torch_value


# TODO: Should this inherit from `nn.Module`?
class MCMC_Optimizer(torch.nn.Module):
    """Markov Chain Monte Carlo optimizer."""

    # pylint: disable=too-many-instance-attributes, abstract-method

    DEFAULT_ROOT_PATH = Path(__file__).parent.parent
    DEFAULT_MODEL_PATH = DEFAULT_ROOT_PATH / "models" / "trRosetta_models"
    DEFAULT_BGND_PATH = DEFAULT_ROOT_PATH / "backgrounds"
    DEFAULT_RESULTS_PATH = DEFAULT_ROOT_PATH / "results"

    def __init__(
        self,
        L,
        aa_weight,
        MCMC,
        native_frequencies,
        experiment_name,
        aa_valid,
        max_aa_index=20,
        sequence_constraint=None,
        target_motif_path=None,
        trRosetta_model_dir="models/trRosetta_models",
        background_distribution_dir="backgrounds",
    ):
        """Construct the optimizer."""
        super().__init__()
        self.results_dir = self.setup_results_dir(experiment_name)
        self.bkg_dir = background_distribution_dir
        self.structure_models = trRosettaEnsemble(
            trRosetta_model_dir
        )  # .share_memory()
        print(
            "%d structure prediction models loaded to %s"
            % (self.structure_models.n_models, d())
        )

        # General params:
        self.eps = 1e-7
        self.seq_L = L

        # Setup MCMC params:
        self.beta, self.N, self.coef, self.M = (
            MCMC["BETA_START"],
            MCMC["N_STEPS"],
            MCMC["COEF"],
            MCMC["M"],
        )
        self.aa_weight = aa_weight

        # Setup sequence constraints:
        self.aa_valid = aa_valid
        self.native_frequencies = native_frequencies

        self.seq_constraint = sequence_constraint
        if self.seq_constraint is not None:
            assert len(self.seq_constraint) == self.seq_L, \
            "Constraint length (%d) must == Seq_L (%d)" %(len(self.seq_constraint), self.seq_L)

            self.seq_constraint = (
                aa2idx(self.seq_constraint).copy().reshape([1, self.seq_L])
            )
            self.seq_constraint_indices = np.where(
                self.seq_constraint != max_aa_index, 1, 0
            )

        self.target_motif_path = target_motif_path
        self.setup_losses()

        # stats
        self.bad_accepts = []
        self.n_accepted_mutations = 0
        self.n_accepted_bad_mutations = 0
        self.best_metrics = {}
        self.best_step = 0
        self.best_sequence = None
        self.best_E = None
        self.step = 0

    def setup_results_dir(self, experiment_name):
        """Create the directories for the results."""
        results_dir = (
            self.DEFAULT_RESULTS_PATH
            / experiment_name
            / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "distogram_evolution").mkdir(parents=True, exist_ok=True)
        print(f"Writing results to {results_dir}")
        return results_dir

    def setup_losses(self):
        """Prepare the loss functions."""

        # Initialize protein background distributions:
        self.bkg_loss = Structural_Background_Loss(self.seq_L, self.bkg_dir)
        self.aa_bkgr_distribution = torch.from_numpy(self.native_frequencies).to(d())

        # Motif-Loss:
        self.motif_weight = 1.00
        self.motif_mask = np.zeros((self.seq_L, self.seq_L))
        self.motif_mask = torch.from_numpy(self.motif_mask).long().to(d())

        if self.target_motif_path is not None:
            self.motif_mask = np.ones((self.seq_L, self.seq_L))
            self.motif_mask = torch.from_numpy(self.motif_mask).long().to(d())
            self.motif_mask.fill_diagonal_(0)
            self.motif_sat_loss = Motif_Satisfaction(
                self.target_motif_path, mask=self.motif_mask, save_dir=self.results_dir
            )

        # Apply the background KL-loss only under the hallucination_mask == 1 region
        self.hallucination_mask = 1 - self.motif_mask
        self.hallucination_mask.fill_diagonal_(0)

    def loss(self, sequence, structure_predictions, msa1hot, track=False):
        """Compute the loss function."""

        # Top-prob:
        TM_score_proxy = top_prob(structure_predictions['dist'], verbose=False)
        TM_score_proxy = TM_score_proxy[0]  # We're running with batch_size = 1

        # Background KL-loss:
        background_loss = self.bkg_loss(
            structure_predictions, hallucination_mask=self.hallucination_mask
        )

        # aa composition loss
        aa_samp = (
            msa1hot[0, :, :20].sum(axis=0) / self.seq_L + self.eps
        )  # Get relative frequency for each AA
        aa_samp = (
            aa_samp / aa_samp.sum()
        )  # Normalize to turn into distributions (possibly redundant)
        loss_aa = (
            aa_samp
            * torch.log(aa_samp / (self.aa_bkgr_distribution + self.eps) + self.eps)
        ).sum()

        # Motif Loss:
        if self.target_motif_path is not None:
            motif_loss = self.motif_sat_loss(structure_predictions)
        else:
            motif_loss = 0

        # total loss
        loss_v = (
            background_loss + self.aa_weight * loss_aa + self.motif_weight * motif_loss
        )

        metrics = {}
        if track:
            metrics["aa_weight"] = self.aa_weight
            metrics["background_loss"] = background_loss
            metrics["total_loss"] = loss_v
            metrics["TM_score_proxy"] = TM_score_proxy

            if self.target_motif_path is not None:
                metrics["motif_loss"] = motif_loss

        return loss_v, metrics

    def metropolis(self, seq, seq_curr, E_curr, E):
        """Compute the Metropolis criterion."""

        # Metropolis criterion
        if E_curr < E:  # Lower energy, replace!
            seq = np.copy(seq_curr)
            E = E_curr
            self.n_accepted_mutations += 1
        else:  # Higher energy, maybe replace..
            if torch.exp((E - E_curr) * self.beta) > np.random.uniform():
                seq = np.copy(seq_curr)
                E = E_curr
                self.bad_accepts.append(1)
                self.n_accepted_bad_mutations += 1
                self.n_accepted_mutations += 1
            else:
                self.bad_accepts.append(0)

        # Update the best sequence:
        if E_curr < self.best_E:
            self.best_E = E_curr
            self.best_sequence = idx2aa(seq_curr[0])
            self.best_step = self.step

        return seq, E

    def mutate(self, seq):
        """Return a mutated version of the sequence."""
        seq_curr = np.copy(seq)

        # Introduce a random mutation using the allowed aa_types:
        idx = np.random.randint(self.seq_L)
        seq_curr[0, idx] = np.random.choice(self.aa_valid)

        if self.seq_constraint is not None:  # Fix the constraint:
            seq_curr = np.where(
                self.seq_constraint_indices, self.seq_constraint, seq_curr
            )

        if np.equal(seq_curr, seq).all():
            # If the mutation did not change anything, retry
            return self.mutate(seq)
        # Otherwise, return the mutated sequence
        return seq_curr

    def fixup_MCMC(self, seq):
        """Dynamically adjust the metropolis beta parameter to improve performance."""
        if self.step - self.best_step > self.N // 4:
            # No improvement for a long time, reload the best_sequence and decrease beta:
            self.best_step = self.step
            self.beta = self.beta / (self.coef ** 2)
            seq = torch.from_numpy(
                aa2idx(self.best_sequence).reshape([1, self.seq_L])
            ).long()

        elif np.mean(self.bad_accepts[-100:]) < 0.05:
            # There has been some progress recently, but we're no longer accepting any bad mutations...
            self.beta = self.beta / self.coef
        else:
            self.beta = self.beta * self.coef

        self.beta = np.clip(self.beta, 5, 200)

        return seq

    @torch.no_grad()
    def run(self, start_seq):
        """Run the MCMC loop."""
        # pylint: disable=too-many-locals

        start_time = time.time()

        # initialize with given input sequence
        print("Initial seq: ", start_seq)
        seq = aa2idx(start_seq).copy().reshape([1, self.seq_L])

        nsave = max(1, self.N // 20)
        E, E_tracker = np.inf, []
        self.bad_accepts = []
        self.n_accepted_mutations = 0
        self.n_accepted_bad_mutations = 0
        self.best_metrics = {}
        self.best_step = 0
        self.best_sequence = start_seq
        self.best_E = E

        # Main loop:
        for self.step in range(self.N + 1):

            # random mutation at random position
            seq_curr = self.mutate(seq)

            # Preprocess the sequence
            seq_curr = torch.from_numpy(seq_curr).long()
            model_input, msa1hot = prep_seq(seq_curr)

            # probe effect of mutation
            structure_predictions = self.structure_models(
                model_input, use_n_models=cfg.n_models
            )
            E_curr, metrics = self.loss(
                seq_curr, structure_predictions, msa1hot, track=True
            )

            seq, E = self.metropolis(seq, seq_curr, E_curr, E)
            E_tracker.append(v(E))

            if self.step % nsave == 0:
                fps = self.step / (time.time() - start_time)
                background_loss = metrics["background_loss"].cpu().detach().numpy()

                print(
                    f"Step {self.step:4d} / {self.N:4d} "
                    f"Loss: {E:.2f}, "
                    f"Bkg-KL: {background_loss:.2f} || "
                    f"beta: {self.beta}, "
                    f"mutations/s: {fps:.2f}, "
                    f"bad_accepts: {np.sum(self.bad_accepts[-100:])}/100",
                    flush=True,
                )

                if self.step % (nsave * 2) == 0:
                    distogram_distribution = (
                        structure_predictions['dist'].detach().cpu().numpy()
                    )
                    distogram = distogram_distribution_to_distogram(
                        distogram_distribution
                    )
                    plot_distogram(
                        distogram,
                        self.results_dir
                        / "distogram_evolution"
                        / f"{self.step:06d}_{E_curr:.4f}.jpg",
                        clim=cfg.limits["dist"],
                    )
                    plot_progress(
                        E_tracker,
                        self.results_dir / "progress.jpg",
                        title=f"Optimization curve after {self.step} steps",
                    )
                    print(f"\n--- Current best: {self.best_sequence}")
                    print(f"--- Structure prediction models: {cfg.n_models}\n")

            if self.step % self.M == 0 and self.step != 0:
                seq = self.fixup_MCMC(seq)

        ########################################

        # Save final results before exiting:
        seq_curr = torch.from_numpy(
            aa2idx(self.best_sequence).reshape([1, self.seq_L])
        ).long()
        model_input, msa1hot = prep_seq(seq_curr)
        structure_predictions = self.structure_models(model_input)
        E_curr, self.best_metrics = self.loss(
            seq_curr, structure_predictions, msa1hot, track=True
        )

        for key in self.best_metrics.keys():
            self.best_metrics[key] = v(metrics[key])
        self.best_metrics["sequence"] = self.best_sequence

        # Dump distogram:
        best_distogram_distribution = structure_predictions['dist'].detach().cpu().numpy()
        distogram = distogram_distribution_to_distogram(best_distogram_distribution)
        plot_distogram(
            distogram,
            self.results_dir / f"{self.best_sequence}.jpg",
            clim=cfg.limits["dist"],
        )
        plot_progress(
            E_tracker,
            self.results_dir / "progress.jpg",
            title=f"Optimization curve after {self.step} steps",
        )

        # Write results to csv:
        with (self.results_dir / "result.csv").open("w") as f:
            for key, val in self.best_metrics.items():
                f.write(f"{key},{val}\n")

        return self.best_metrics
