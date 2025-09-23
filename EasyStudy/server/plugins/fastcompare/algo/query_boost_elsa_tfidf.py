import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)

from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k, hidden_dim

import logging
from datetime import datetime
import os

# Create log directory and file handler for detailed logging
log_dir = "c:\\Users\\elisk\\Desktop\\2024-25\\LS\\Recommenders\\logs"
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(
    log_dir, f"query_boost_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

print(f"Query boost logging to: {log_filename}")


class QueryBoostELSAtfidf(AlgorithmBase):
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
        return "ELSA+SAE Query tf-idf"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "alpha",
                ParameterType.FLOAT,
                0.7,
                help="Weight of user feedback vs. query boost",
            )
        ]

    def set_query(self, query):
        self.query = query or ""
        logger.info(f"Query set to: '{self.query}'")

    def fit(self):
        logger.info("Loading ELSA+SAE models...")
        num_items = self.num_items
        self.elsa = ELSA(num_items, latent_dim)
        state = torch.load("models/elsa_model_best.pt")
        loaded_A = state.get("A")
        if loaded_A is not None and loaded_A.shape[0] != num_items:
            raise RuntimeError(
                f"Pretrained ELSA model expects {loaded_A.shape[0]} items but the current dataset has {num_items}. "
                "Use a model trained on the same dataset."
            )
        self.elsa.load_state_dict(state)
        self.sae = TopKSAE(latent_dim, hidden_dim, k)
        self.sae.load_state_dict(torch.load("models/sae_model_r4_k32.pt"))
        data = torch.load("models/tag_neuron_map_tf_idf.pt")
        self.tag_tensor = data["tag_tensor"]
        emb = torch.load("models/tag_embeddings_tf_idf.pt")
        self.unique_tags = emb["unique_tags"]
        self.tag_embeddings = emb["embeddings"]
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Models loaded successfully")

    def _query_to_vector(self, query, top_n=5):
        if not query:
            logger.info("No query provided, skipping query boosting")
            return None

        logger.info(f"Processing query: '{query}'")

        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_emb, self.tag_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_n)
        indices = top_results.indices
        scores = top_results.values

        # Log the top matching tags
        logger.info(f"Top {top_n} matching tags for query '{query}':")
        for i, (idx, score) in enumerate(zip(indices, scores)):
            tag_name = self.unique_tags[idx.item()]
            logger.info(f"  {i+1}. '{tag_name}' (similarity: {score.item():.4f})")

        total = scores.sum()
        weights = scores / total if total > 0 else torch.ones_like(scores) / len(scores)

        logger.info(f"Tag weights: {[f'{w.item():.4f}' for w in weights]}")

        weighted_sum = torch.zeros(
            self.tag_tensor.shape[1], dtype=self.tag_tensor.dtype
        )
        for idx, weight in zip(indices, weights):
            weighted_sum += weight * self.tag_tensor[idx]

        # Log some statistics about the boost vector
        boost_stats = {
            "mean": weighted_sum.mean().item(),
            "std": weighted_sum.std().item(),
            "max": weighted_sum.max().item(),
            "min": weighted_sum.min().item(),
            "nonzero_count": (weighted_sum != 0).sum().item(),
        }
        logger.info(f"Query boost vector stats: {boost_stats}")

        return weighted_sum

    def predict(self, selected_items, filter_out_items, k):
        print(f"Using predict query: {self.query}")
        logger.info(
            f"Predicting with {len(selected_items)} selected items, alpha={self.alpha}"
        )

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

            # Log original sparse representation stats
            original_stats = {
                "mean": h_sparse.mean().item(),
                "std": h_sparse.std().item(),
                "max": h_sparse.max().item(),
                "nonzero_count": (h_sparse != 0).sum().item(),
            }
            logger.info(f"Original sparse representation stats: {original_stats}")

            boost = self._query_to_vector(self.query)

            if boost is not None:
                boost_unsqueezed = boost.unsqueeze(0)
                h_sparse_original = h_sparse.clone()
                h_sparse = self.alpha * h_sparse + (1 - self.alpha) * boost_unsqueezed

                # Log the effect of boosting
                boost_effect = torch.abs(h_sparse - h_sparse_original).mean().item()
                logger.info(
                    f"Query boost effect (mean absolute change): {boost_effect:.6f}"
                )

                # Log boosted representation stats
                boosted_stats = {
                    "mean": h_sparse.mean().item(),
                    "std": h_sparse.std().item(),
                    "max": h_sparse.max().item(),
                    "nonzero_count": (h_sparse != 0).sum().item(),
                }
                logger.info(f"Boosted sparse representation stats: {boosted_stats}")
            else:
                logger.info("No query boost applied")

            recon_z = self.sae.dec(h_sparse)
            scores = torch.matmul(recon_z, self.elsa.A.T) - user_vector.unsqueeze(0)

        scores_np = scores.squeeze().numpy()

        # Log score statistics before filtering
        logger.info(
            f"Recommendation scores stats: mean={scores_np.mean():.4f}, "
            f"std={scores_np.std():.4f}, max={scores_np.max():.4f}, min={scores_np.min():.4f}"
        )

        mask = np.isin(np.arange(self.num_items), filter_out_items)
        scores_np[mask] = -np.inf
        top_indices = np.argsort(-scores_np)[:k]

        # Log top recommendations with scores
        logger.info(f"Top {min(5, k)} recommendations:")
        for i, idx in enumerate(top_indices[:5]):
            score = scores.squeeze()[idx].item()
            # You might want to add item title lookup here if available
            logger.info(f"  {i+1}. Item {idx} (score: {score:.4f})")

        return top_indices.tolist()
