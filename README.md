# trdesign-pytorch
This repository is a PyTorch implementation of the [trDesign paper.](https://www.biorxiv.org/content/10.1101/2020.07.22.211482v1.full.pdf)
The official TensorFlow implementation is [here.](https://github.com/gjoni/trDesign)

![Alt text](assets/trDesign.jpg?raw=true "Title")

The initial port of the trRosetta network was done by [@lucidrains](https://github.com/lucidrains).
## Usage (protein structure prediction):
simply run:
- cd src/
- python predict.py path_to_some_a3m_file.a3m
or 
- python predict.py path_to_some_fasta_file.fasta

## Usage (protein design):
1. edit src/config.py to set the experiment configuration
2. run python run.py
3. All results will be saved under results/

## Configuration options:
- Sequence length (int)
- AA_weight (float): how strongly we want the amino acid type composition to be 'natural'
- RM_AA (string): disable specific amino acid types
- n_models (int): how many trRosetta model ensembles we want to use during the MCMC loop
- sequence constraint: fix a subset of the sequence residues to specific amino acids
- target_motif: optimize a sequence with a target motif provided as a .npz file
- MCMC options
