#!/usr/bin/env python
"""Structure prediction.

Based on the implementation by @lucidrains:
https://github.com/lucidrains/tr-rosetta-pytorch/blob/main/tr_rosetta_pytorch/cli.py
"""

# native
from inspect import cleandoc
from pathlib import Path
import sys

# lib
import numpy as np
import torch

# pkg
from tr_Rosetta_model import trRosettaEnsemble, preprocess
import utils


def get_ensembled_predictions(input_file, output_file=None):
    """Use an ensemble of pre-trained networks to predict the structure of an MSA file."""
    ensemble_model = trRosettaEnsemble()

    input_path = Path(input_file)
    input_data, _ = preprocess(msa_file=input_path)
    # input_data, _ = preprocess(use_random_seq = True)

    output_path = (
        Path(output_file)
        if output_file
        else input_path.parent / f"{input_path.stem}.npz"
    )
    # prob_theta, prob_phi, prob_distance, prob_omega
    outputs = [model(input_data) for model in ensemble_model.models]
    averaged_outputs = [
        torch.stack(model_output).mean(dim=0).cpu().detach().numpy()
        for model_output in zip(*outputs)
    ]
    output_dict = dict(zip(["theta", "phi", "dist", "omega"], averaged_outputs))
    np.savez_compressed(output_path, **output_dict)
    print(f"predictions for {input_path} saved to {output_path}")

    utils.plot_distogram(
        utils.distogram_distribution_to_distogram(output_dict["dist"]),
        f"{input_file}_dist.jpg",
    )


def main():
    """Predict structure using an ensemble of models.

    Usage: predict.py <input> [<output>]

    Options:
        <input>                 input `.a3m` or `.fasta` file
        <output>                output file (by default adds `.npz` to <input>)

    Examples:

    $ ./predict.py data/test.a3m
    $ ./predict.py data/test.fasta
    """
    show_usage = False
    args = sys.argv[1:]
    if len(args) == 1 and args[0] in ["-h", "--help"]:
        show_usage = True
    if not 1 <= len(args) <= 2:
        show_usage = True
        print("ERROR: Unknown number of arguments.\n\n")

    if show_usage:
        print(f"{cleandoc(main.__doc__)}\n")
        sys.exit(1)

    get_ensembled_predictions(*args)  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
