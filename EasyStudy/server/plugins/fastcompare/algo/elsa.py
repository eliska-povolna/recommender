import torch
import numpy as np

from plugins.fastcompare.algo.algorithm_base import AlgorithmBase
from train_elsa import ELSA, latent_dim

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
        self.model = ELSA(self.num_items, latent_dim)
        self.model.load_state_dict(torch.load("models/elsa_model.pt"))

    def predict(self, selected_items, filter_out_items, k):
        user_vector = torch.zeros(self.num_items, dtype=torch.float32)
        for i in selected_items:
            user_vector[i] = 1.0
        with torch.no_grad():
            A = torch.nn.functional.normalize(self.model.A, dim=-1)
            z = torch.matmul(user_vector.unsqueeze(0), A)
            scores = torch.matmul(z, A.T) - user_vector.unsqueeze(0)
        scores_np = scores.squeeze().numpy()
        mask = np.isin(np.arange(self.num_items), filter_out_items)
        scores_np[mask] = -np.inf
        top_indices = np.argsort(-scores_np)[:k]
        return top_indices.tolist()
