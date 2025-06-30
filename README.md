# Recommender

This repository contains a small movie recommendation pipeline based on the MovieLens dataset. It provides scripts for preprocessing the data, training recommendation models, mapping neurons to textual tags and running interactive inference.

## Setup

1. Download the MovieLens `ml-latest` dataset and extract it into the `data/` directory so that files like `ratings.csv` and `movies.csv` are present.
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create an empty `models/` directory where trained models will be saved.

## Training

Run the following scripts in order:

1. **Preprocess** the data and create train/test splits:
   ```bash
   python preprocess.py
   ```
2. **Train the ELSA collaborative filtering model:**
   ```bash
   python train_elsa.py
   ```
3. **Train the sparse autoencoder on ELSA embeddings:**
   ```bash
   python train_sae.py
   ```
4. **Map neurons to textual tags and compute tag embeddings:**
   ```bash
   python map_neurons_to_tags.py
   python embeddings.py
   ```

## Evaluation

After training you can evaluate the recommender on held-out users:
```bash
python evaluate.py
```

## Inference

`inference.py` provides an interactive interface that accepts a user index and a free-text query describing desired movie properties. The query is converted to a boosting vector over hidden neurons and used to generate recommendations.

```bash
python inference.py
```

The script prints a list of recommended movie titles with their scores.

## Repository Structure

- `preprocess.py` – prepares the data set and builds sparse matrices
- `train_elsa.py` – trains the collaborative filtering model (ELSA)
- `train_sae.py` – trains the sparse autoencoder
- `map_neurons_to_tags.py` – aggregates activations of hidden neurons by tags
- `embeddings.py` – computes sentence-transformer embeddings for tags
- `evaluate.py` – evaluates Recall@20 and NDCG@20 on test users
- `inference.py` – interactive demo for querying recommendations

Feel free to adapt the code for your experiments.
