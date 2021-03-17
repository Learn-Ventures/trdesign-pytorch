# trdesign-pytorch
This repository is a PyTorch implementation of the [trDesign paper][1].
The official TensorFlow implementation is [here](https://github.com/gjoni/trDesign). The initial port of the trRosetta network was done by [@lucidrains](https://github.com/lucidrains).

![Figure 1: trDesign Architecture](assets/trDesign.jpg?raw=true "trDesign Architecture")

<small>Figure 1 of <em>De novo protein design by deep network hallucination</em> (p. 12, [Anishchenko et al.][1], [CC-BY-ND][cc-by-nd])</small>

[1]: https://www.biorxiv.org/content/10.1101/2020.07.22.211482v1.full.pdf
[cc-by-nd]: https://creativecommons.org/licenses/by-nc-nd/4.0/

## Requirements
Requires python 3.8+.

```bash
pip install matplotlib numpy torch
```

## Usage (protein structure prediction):
Simply run:
```bash
cd src
python predict.py path_to_some_a3m_file.a3m
# or
python predict.py path_to_some_fasta_file.fasta
```

## Usage (protein design):
1. edit `src/config.py` to set the experiment configuration
2. run `python run.py`
3. All results will be saved under `results/`

## Configuration options:
- Sequence length (`int`)
- AA_weight (`float`): how strongly we want the amino acid type composition to be 'natural'
- RM_AA (`str`): disable specific amino acid types
- n_models (`int`): how many `trRosetta` model ensembles we want to use during the MCMC loop
- sequence constraint (`str`): fix a subset of the sequence residues to specific amino acids
- target_motif (`path`): optimize a sequence with a target motif provided as an `.npz` file
- MCMC options
