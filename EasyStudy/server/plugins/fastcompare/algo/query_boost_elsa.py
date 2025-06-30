import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType

from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k, hidden_dim

class QueryBoostELSA(AlgorithmBase):
    """ELSA + SAE recommender allowing query-based boosting."""

    def __init__(self, loader, alpha=0.7, **kwargs):
        self.loader = loader
        self.alpha = float(alpha)
        self.query = ""
        self.num_items = loader.items_df.shape[0]
        self.elsa = None
        self.sae = None
        self.embed_model = None
        self.tag_tensor = None
        self.tag_embeddings = None
        self.unique_tags = None

    @classmethod
    def name(cls):
        return "ELSA+SAE Query"

    @classmethod
    def parameters(cls):
        return [
            Parameter("alpha", ParameterType.FLOAT, 0.7, help="Weight of user feedback vs. query boost")
        ]

    def set_query(self, query):
        self.query = query or ""

    def fit(self):
        num_items = self.num_items
        self.elsa = ELSA(num_items, latent_dim)
        self.elsa.load_state_dict(torch.load("models/elsa_model.pt"))
        self.sae = TopKSAE(latent_dim, hidden_dim, k)
        self.sae.load_state_dict(torch.load("models/sae_model.pt"))
        data = torch.load("models/tag_neuron_map.pt")
        self.tag_tensor = data["tag_tensor"]
        emb = torch.load("models/tag_embeddings.pt")
        self.unique_tags = emb["unique_tags"]
        self.tag_embeddings = emb["embeddings"]
        self.embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def _query_to_vector(self, query, top_n=5):
        if not query:
            return None
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_emb, self.tag_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_n)
        indices = top_results.indices
        scores = top_results.values
        total = scores.sum()
        weights = scores / total if total > 0 else torch.ones_like(scores) / len(scores)
        weighted_sum = torch.zeros(self.tag_tensor.shape[1], dtype=self.tag_tensor.dtype)
        for idx, weight in zip(indices, weights):
            weighted_sum += weight * self.tag_tensor[idx]
        return weighted_sum

    def predict(self, selected_items, filter_out_items, k):
        user_vector = torch.zeros(self.num_items, dtype=torch.float32)
        for i in selected_items:
            user_vector[i] = 1.0
        with torch.no_grad():
            z = torch.matmul(user_vector.unsqueeze(0), self.elsa.A)
            h = torch.relu(self.sae.enc(z))
            h_sparse = torch.where(
                h >= torch.topk(h, self.sae.k, dim=1).values[:, -1].unsqueeze(1),
                h,
                torch.zeros_like(h),
            )
            boost = self._query_to_vector(self.query)
            if boost is not None:
                h_sparse = self.alpha * h_sparse + (1 - self.alpha) * boost.unsqueeze(0)
            recon_z = self.sae.dec(h_sparse)
            scores = torch.matmul(recon_z, self.elsa.A.T) - user_vector.unsqueeze(0)
        scores_np = scores.squeeze().numpy()
        mask = np.isin(np.arange(self.num_items), filter_out_items)
        scores_np[mask] = -np.inf
        top_indices = np.argsort(-scores_np)[:k]
        return top_indices.tolist()
