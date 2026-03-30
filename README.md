# MiGu and NaNa: Semantic Data Augmentation for Protein Classification in Graph Neural Networks

This repository contains the implementation of **MiGu** and **NaNa**, two semantic data augmentation strategies for **protein classification with graph neural networks (GNNs)**.

This project is part of my **first-author research work**, and I was responsible for the **full pipeline**, including:
- model design and implementation
- structural and biophysical feature engineering
- semantic augmentation design
- graph dataset construction
- experiment setup and evaluation

---

## Overview

Protein classification with graph neural networks can be limited by incomplete structural representation and insufficient training diversity.  
To address this, this project introduces semantic augmentation strategies that enrich protein graph representations with additional structural and biophysical information.

This repository includes:
- structure-based feature extraction
- graph construction from protein structural data
- semantic augmentation methods
- graph neural network model implementation
- training and evaluation pipeline for protein classification

The experiments are based on the **EC** and **SCOPe** datasets.

---

## Repository Structure

- `hydrogen_bond.py`  
  Generates structural and biophysical features from protein structures and constructs graph-formatted datasets for downstream learning.

- `graph_for_MIGU_NANA.py`  
  Main training and evaluation pipeline for protein classification experiments, including augmented (`mine`) and baseline (`their`) settings.

- `backbone_GIN_backup.py`  
  Implementation of the graph neural network backbone with geometric and structural feature integration.

- `environment_pygdemo.yml`  
  Conda environment for model training and graph learning experiments.

- `environment_bio.yml`  
  Conda environment for structure processing and biophysical feature generation.

---

## Datasets

This project uses the **EC** and **SCOPe** protein classification datasets.

The datasets are **not included in this repository**, because they are obtained from external sources and are not redistributed here.  
Please download them from their original sources and place them in the appropriate local directory before running the code.

You may then specify the dataset location using the `--dataset_path` argument in the training script.

---

## Environment Setup

### Training environment
```bash
conda env update --file environment_pygdemo.yml
conda activate MIGU
```

### Feature generation environment
```bash
conda env update --file environment_bio.yml
conda activate hydrogen_bond
```

---

## Workflow

### Step 1: Generate structural and biophysical features
```bash
python hydrogen_bond.py
```

### Step 2: Train the model

Example command:

```bash
CUDA_LAUNCH_BLOCKING=1 python graph_for_MIGU_NANA.py \
  --training_title 226_MPNN_true_edge \
  --epochs 4000 \
  --lr 0.0005 \
  --optimizer adam \
  --dim_h 128 \
  --batch_size 50 \
  --model GIN_Attribute \
  --num_workers 8 \
  --dataset_path /path/to/dataset \
  --dataset EC \
  --edge true \
  --eval_batch_size 20 \
  --version mine \
  --type GCN \
  --bond false
```

---

## Command Line Arguments

- `--training_title`: experiment name
- `--epochs`: number of training epochs
- `--lr`: learning rate
- `--optimizer`: optimizer type
- `--dim_h`: hidden dimension size
- `--batch_size`: training batch size
- `--model`: model type
- `--num_workers`: number of dataloader workers
- `--dataset_path`: path to dataset directory
- `--dataset`: dataset name
- `--edge`: whether to use co-embedding residual learning
- `--eval_batch_size`: evaluation batch size
- `--version`: `mine` for augmented setting, `their` for baseline setting
- `--type`: layer type (`GCN`, `GIN`, `MPNN`)
- `--bond`: whether to use edge attributes; if true, MiGu augmentation is used, otherwise NaNa augmentation is used

---

## Notes

- `mine` indicates the augmented setting.
- `their` indicates the original or non-augmented baseline setting.
- Some paths in the current scripts are hard-coded and may need to be adapted to your local directory structure before running.
- This repository focuses on the implementation pipeline; dataset redistribution is not included.

---

## Contribution

This repository corresponds to my **first-author work**, and I developed the project end-to-end, including:
- structural feature extraction
- biophysical feature generation
- graph construction
- augmentation framework design
- model implementation
- experiment execution and evaluation

---

## Citation

If you use this repository, please cite the associated paper once available.
