# Recommender

This project contains scripts for building a recommendation system using a Collaborative Filtering Autoencoder (CFAE) and a Sparse Autoencoder (SAE). The data is based on the MovieLens dataset stored in `data/`.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Workflow

1. Run `preprocess.py` to create train and test matrices.
2. Train the CFAE model with `train_elsa.py`.
3. Train the sparse autoencoder with `train_sae.py`.
4. Map SAE neurons to movie tags using `map_neurons_to_tags.py`.
5. Generate recommendations via `inference.py`.

All scripts expect the preprocessed `.pkl` files to be present in the repository root.
