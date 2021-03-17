# trdesign-pytorch

This repository is a PyTorch implementation of the [trDesign paper][1] based on the
[official TensorFlow implementation](https://github.com/gjoni/trDesign). The initial port of the trRosetta network was done by [@lucidrains](https://github.com/lucidrains).

![Figure 1: trDesign Architecture](assets/trDesign.jpg?raw=true "trDesign Architecture")

<small>Figure 1 of <em>De novo protein design by deep network hallucination</em> (p. 12, [Anishchenko et al.][1], [CC-BY-ND][cc-by-nd])</small>

[1]: https://www.biorxiv.org/content/10.1101/2020.07.22.211482v1.full.pdf
[cc-by-nd]: https://creativecommons.org/licenses/by-nc-nd/4.0/

## Requirements

Requires python 3.6+

```bash
pip install matplotlib numpy torch
```

## Usage (protein structure prediction):

Simply run:

```bash
python predict.py path_to_a3m_file.a3m
# or
python predict.py path_to_fasta_file.fasta
```

## Usage (protein design):

1. edit `src/config.py` to set the experiment configuration
2. run `python design.py`
3. All results will be saved under `results/`

## Configuration options:

- Sequence length (`int`)
- AA_weight (`float`): how strongly we want the amino acid type composition to be 'natural'
- RM_AA (`str`): disable specific amino acid types
- n_models (`int`): how many `trRosetta` model ensembles we want to use during the MCMC loop
- sequence constraint (`str`): fix a subset of the sequence residues to specific amino acids
- target_motif (`path`): optimize a sequence with a target motif provided as an `.npz` file
- MCMC options

## References

```bibtex
@article {Yang1496,
	author = {Yang, Jianyi and Anishchenko, Ivan and Park, Hahnbeom and Peng, Zhenling and Ovchinnikov, Sergey and Baker, David},
	title = {Improved protein structure prediction using predicted interresidue orientations},
	year = {2020},
	doi = {10.1073/pnas.1914677117},
	URL = {https://www.pnas.org/content/117/3/1496},
	eprint = {https://www.pnas.org/content/117/3/1496.full.pdf},
	journal = {Proceedings of the National Academy of Sciences}
}
```

```bibtex
@article {Anishchenko2020.07.22.211482,
	author = {Anishchenko, Ivan and Chidyausiku, Tamuka M. and Ovchinnikov, Sergey and Pellock, Samuel J. and Baker, David},
	title = {De novo protein design by deep network hallucination},
	year = {2020},
	doi = {10.1101/2020.07.22.211482},
	URL = {https://www.biorxiv.org/content/early/2020/07/23/2020.07.22.211482},
	eprint = {https://www.biorxiv.org/content/early/2020/07/23/2020.07.22.211482.full.pdf},
	journal = {bioRxiv}
}
```

```bibtex
@article {Tischer2020.11.29.402743,
	author = {Tischer, Doug and Lisanza, Sidney and Wang, Jue and Dong, Runze and Anishchenko, Ivan and Milles, Lukas F. and Ovchinnikov, Sergey and Baker, David},
	title = {Design of proteins presenting discontinuous functional sites using deep learning},
	year = {2020},
	doi = {10.1101/2020.11.29.402743},
	URL = {https://www.biorxiv.org/content/early/2020/11/29/2020.11.29.402743},
	eprint = {https://www.biorxiv.org/content/early/2020/11/29/2020.11.29.402743.full.pdf},
	journal = {bioRxiv}
}
```
