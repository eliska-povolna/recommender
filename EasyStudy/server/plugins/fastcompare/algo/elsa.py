import torch
import numpy as np

from plugins.fastcompare.algo.algorithm_base import AlgorithmBase
from train_elsa import ELSA, latent_dim

import pickle


class ELSAAlgorithm(AlgorithmBase):
    """Plain ELSA recommender loading pretrained weights."""

    def __init__(self, loader, **kwargs):
        self.loader = loader
        self.num_items = loader.items_df.shape[0]
        self.model = None

    @classmethod
    def name(cls):
        return "ELSA"

    @classmethod
    def parameters(cls):
        return []

    def fit(self):
        # Load your full trained model
        state = torch.load("models/elsa_model_best.pt")
        model_size = state["A"].shape[0]  # Get actual size

        full_model = ELSA(model_size, latent_dim)
        full_model.load_state_dict(state)

        # Create mapping between server items and trained model items
        server_movie_ids = self.loader.items_df["item_id"].values

        # Load the mapping used during training
        with open("data/item2index.pkl", "rb") as f:
            training_item2index = pickle.load(f)

        # Extract embeddings for server items only
        server_embeddings = []
        valid_indices = []

        for i, movie_id in enumerate(server_movie_ids):
            if movie_id in training_item2index:
                training_idx = training_item2index[movie_id]
                server_embeddings.append(full_model.A[training_idx])
                valid_indices.append(i)

        if not server_embeddings:
            raise RuntimeError("No overlap between server movies and training data!")

        # Create smaller model with extracted embeddings
        self.model = ELSA(len(valid_indices), latent_dim)
        self.model.A.data = torch.stack(server_embeddings)

        # Store the mapping instead of modifying the loader
        self.valid_indices = (
            valid_indices  # Map from our model indices to original loader indices
        )
        self.num_items = len(valid_indices)

        print(f"✅ Extracted {self.num_items} items from {model_size}-item model")

    def predict(self, selected_items, filter_out_items, k):
        user_vector = torch.zeros(self.num_items, dtype=torch.float32)
        for i in selected_items:
            # selected_items are indices in the original loader space
            # We need to map them to our extracted model space
            if i in self.valid_indices:
                model_idx = self.valid_indices.index(i)
                user_vector[model_idx] = 1.0

        with torch.no_grad():
            A = torch.nn.functional.normalize(self.model.A, dim=-1)
            z = torch.matmul(user_vector.unsqueeze(0), A)
            scores = torch.matmul(z, A.T) - user_vector.unsqueeze(0)

        scores_np = scores.squeeze().numpy()

        # Map filter_out_items from loader space to model space
        model_filter_indices = []
        for item_idx in filter_out_items:
            if item_idx in self.valid_indices:
                model_idx = self.valid_indices.index(item_idx)
                model_filter_indices.append(model_idx)

        if model_filter_indices:
            scores_np[model_filter_indices] = -np.inf

        top_model_indices = np.argsort(-scores_np)[:k]

        # Map back from model space to loader space
        top_loader_indices = [self.valid_indices[idx] for idx in top_model_indices]

        return top_loader_indices
