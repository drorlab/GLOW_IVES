# GLOW-IVES

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

----------------------------------

## Overview

This repository provides a Python implementation of two improved protocols for pose sampling: GLOW (auGmented sampLing with sOftened vdW potential) and IVES (IteratiVe Ensemble Sampling). For more details on the protocols, refer to [this link](https://arxiv.org/abs/2312.00191). In addition, we provided new cross-docking datasets generated using GLOW and IVES containing approximately 5,000 protein-ligand cross-docking pairs, serving as invaluable resources for training and evaluating machine learning-based scoring functions. The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1CMlsJzDzIJNSsWbcLDhs4JT-VpaBChOJ).

## Installation

```
conda create -n glow_ives python=3.10 pip -y
conda activate glow_ives
conda install -y -c conda-forge openmm==7.7.0
conda install -y -c conda-forge openff-toolkit==0.14.0
conda install -y -c conda-forge openmmforcefields==0.11.2
conda install -y -c omnia pdbfixer==1.8.1
conda install -y -c conda-forge openbabel rdkit smina
pip install click tqdm biopandas==0.4.1 easy-parallel
```

## Usage

To run GLOW, run the following command:

`python -m glow_ives.sampling.glow glow <OUTPUT_PREFIX> <DOCKING_PROTEIN_STRUCTURE_PDBQT> <LIGAND_TO_DOCK_SDF> <REF_LIGAND_FOR_BOX_CENTER_SDF>`

To run IVES, run the following command:

`# python -m glow_ives.sampling.ives run_ives <OUTPUT_DIR> <DOCKING_PROTEIN_STRUCTURE_PDB> <DOCKING_PROTEIN_STRUCTURE_PDBQT> <LIGAND_TO_DOCK_SDF> <REF_LIGAND_FOR_BOX_CENTER_SDF> --num_conformations <NUM_OF_PROTEIN_CONFORMATIONS> --num_cpus <NUM_OF_CPUS> --num_iter <NUM_OF_ITERATIONS>`
