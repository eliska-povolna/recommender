# Recommender

This repository contains a small movie recommendation pipeline based on the MovieLens dataset. It provides scripts for preprocessing the data, training recommendation models, mapping neurons to textual tags and running interactive inference.

## Setup

1. Download the MovieLens `ml-latest` dataset and extract it into the `data/` directory so that files like `ratings.csv` and `movies.csv` are present.
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create an empty `models/` directory where trained models will be saved.
   Any pretrained `.pt` files used by the EasyStudy plugin should also be
   stored in this folder. **The models must be trained on the exact dataset
   that is loaded inside EasyStudy**. Otherwise the loader will raise an error
   due to mismatching weight shapes.

If you want to train the models on exactly the same data that the EasyStudy
plugin uses internally, run:

```bash
python export_easystudy_dataset.py
```

This script loads the filtered MovieLens dataset using the same preprocessing
steps as the `fastcompare` plugin and writes `data/processed_train.pkl`,
`data/processed_test.pkl` and `data/item2index.pkl` for the training scripts
below.

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
