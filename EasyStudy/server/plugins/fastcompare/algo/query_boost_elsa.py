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
import pickle

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


class QueryBoostELSA(AlgorithmBase):
    """ELSA + SAE recommender allowing query-based boosting."""

    def __init__(self, loader, alpha=0.3, **kwargs):
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

        # Optional eager load of embeddings (kept for compatibility)
        emb = torch.load("models/tag_embeddings_centroid.pt", weights_only=False)
        self.tag_emb = torch.tensor(emb["embeddings"], dtype=torch.float32)
        self.tag_texts = emb.get("processed_tags", emb.get("unique_tags"))
        self.encoder_model = SentenceTransformer(
            emb.get("model_name", "sentence-transformers/all-distilroberta-v1")
        )

    @classmethod
    def name(cls):
        return "ELSA+SAE Query"

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

    def _ensure_loaded(self):
        """Lazy-load models and mappings if not loaded yet."""
        need_elsa = self.elsa is None
        need_sae = self.sae is None
        need_tags = not hasattr(self, "tag_mapping") or self.tag_mapping is None
        if need_elsa or need_sae or need_tags:
            self.fit()

    def set_query(self, query):
        self.query = query or ""
        logger.info(f"Query set to: '{self.query}'")

    def fit(self):
        logger.info("Loading ELSA+SAE models...")

        # Load full trained model first
        state = torch.load("models/elsa_model_best.pt", weights_only=False)
        model_size = state["A"].shape[0]  # Get actual size dynamically

        full_elsa = ELSA(model_size, latent_dim)  # Use actual size
        full_elsa.load_state_dict(state)

        # Load training item mapping
        with open("data/item2index.pkl", "rb") as f:
            training_item2index = pickle.load(f)

        # Get server movie IDs
        server_movie_ids = self.loader.items_df["item_id"].values

        # Extract embeddings for server items only
        server_embeddings = []
        valid_server_indices = []

        for i, movie_id in enumerate(server_movie_ids):
            if movie_id in training_item2index:
                training_idx = training_item2index[movie_id]
                server_embeddings.append(full_elsa.A[training_idx])
                valid_server_indices.append(i)

        if not server_embeddings:
            raise RuntimeError("No overlap between server movies and training data!")

        # Create smaller ELSA model with extracted embeddings
        self.elsa = ELSA(len(valid_server_indices), latent_dim)
        self.elsa.A.data = torch.stack(server_embeddings)

        # Store the mapping instead of modifying the loader
        self.valid_indices = valid_server_indices  # Map from model to loader indices
        self.num_items = len(valid_server_indices)

        logger.info(f"Extracted {self.num_items} items from {model_size}-item model")

        # Rest of the method stays the same...
        self.A_norm = torch.nn.functional.normalize(self.elsa.A, dim=-1)

        # Load SAE
        self.sae = TopKSAE(latent_dim, hidden_dim, k)
        self.sae.load_state_dict(
            torch.load("models/sae_model_r4_k32.pt", weights_only=False)
        )

        # Load tag mapping data
        tag_data = torch.load("models/tag_neuron_map_centroid.pt", weights_only=False)
        self.tag_mapping = {
            "tag_tensor": tag_data["tag_tensor"],  # [T, H]
            "unique_tags": tag_data["unique_tags"],
        }
        assert (
            self.tag_mapping["tag_tensor"].shape[1] == hidden_dim
        ), f"Tag tensor H={self.tag_mapping['tag_tensor'].shape[1]} != hidden_dim={hidden_dim}"

        # Load embeddings for sentence transformer
        emb = torch.load("models/tag_embeddings_centroid.pt", weights_only=False)
        self.tag_embeddings = emb["embeddings"]  # [T, D]
        self.tag_texts = emb.get("processed_tags", emb.get("unique_tags"))
        self.sentence_model = SentenceTransformer(
            emb.get("model_name", "sentence-transformers/all-distilroberta-v1")
        )

        logger.info("Models loaded successfully")
        logger.info(f"Loaded {len(self.tag_mapping['unique_tags'])} tag mappings")

    def _query_to_vector(self, query):
        self._ensure_loaded()
        if not hasattr(self, "tag_mapping") or self.tag_mapping is None:
            logger.warning("Tag mapping not loaded")
            return None

        tag_tensor = self.tag_mapping["tag_tensor"]  # [T, H]
        tag_emb = torch.as_tensor(self.tag_embeddings, dtype=torch.float32)  # [T, D]

        # PŘIDEJ DEBUGGING:
        logger.info(f"DEBUG Tag Mapping Stats:")
        logger.info(f"  Tag tensor shape: {tag_tensor.shape}")
        logger.info(f"  Tag tensor mean: {tag_tensor.mean():.6f}")
        logger.info(f"  Tag tensor max: {tag_tensor.max():.6f}")
        logger.info(f"  Tag tensor nonzero: {(tag_tensor > 1e-6).sum().item()}")

        # query → embedding
        q = self.sentence_model.encode(query, convert_to_tensor=True)  # [D]
        sim = util.cos_sim(q, tag_emb).squeeze(0)  # [T]

        topk = min(10, sim.shape[0])
        vals, idx = torch.topk(sim, k=topk, largest=True)

        interpreted_tags = [self.tag_texts[t] for t in idx.tolist()]
        logger.info(
            f"Query '{query}' interpreted as tags: {interpreted_tags} "
            f"(scores: {[round(v.item(), 4) for v in vals]})"
        )

        # temperature-softmax mixing
        w = torch.softmax(vals / 0.3, dim=0)

        # PŘIDEJ DEBUGGING WEIGHTS:
        logger.info(f"Softmax weights: {[round(weight.item(), 4) for weight in w]}")

        boost = torch.zeros(tag_tensor.shape[1], dtype=torch.float32)
        for j, t in enumerate(idx.tolist()):
            contribution = w[j] * tag_tensor[t]
            boost += contribution
            # LOG TOP 3 CONTRIBUTIONS:
            if j < 3:
                logger.info(
                    f"  Tag '{interpreted_tags[j]}': weight={w[j]:.4f}, "
                    f"tag_vector_norm={tag_tensor[t].norm():.4f}, "
                    f"contribution_norm={contribution.norm():.4f}"
                )

        # FINAL BOOST STATS:
        logger.info(
            f"Final boost vector: norm={boost.norm():.4f}, "
            f"max={boost.max():.4f}, nonzero={((boost > 1e-6).sum().item())}"
        )

        if boost.norm() == 0:
            logger.warning("Boost vector has zero norm")
            return None
        return boost

    def _get_tensor_stats(self, tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max": tensor.max().item(),
            "nonzero_count": (tensor > 0).sum().item(),
        }

    def predict(self, selected_items, filter_out_items, k):
        self._ensure_loaded()
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

            # Apply query boost if query exists
            if self.query:
                logger.info(f"Applying query boost for query: '{self.query}'")
                boost = self._query_to_vector(self.query)
                logger.info(f"Boost vector stats: {self._get_tensor_stats(boost)}")
                if boost is not None:
                    boost_unsqueezed = boost.unsqueeze(0)
                    h_sparse_original = h_sparse.clone()

                    # PŘED BOOST - výpočet scores pro analýzu:
                    recon_z_before = self.sae.dec(h_sparse_original)
                    A_norm = torch.nn.functional.normalize(self.elsa.A, dim=-1)
                    recon_z_norm_before = torch.nn.functional.normalize(
                        recon_z_before, dim=-1
                    )
                    scores_before = torch.matmul(
                        recon_z_norm_before, A_norm.T
                    ) - user_vector.unsqueeze(0)
                    scores_before_np = scores_before.squeeze().numpy()

                    # APLIKUJ BOOST:
                    h_sparse = (
                        self.alpha * h_sparse + (1 - self.alpha) * boost_unsqueezed * 50
                    )
                    k_relaxed = min(int(self.sae.k * 1.5), h_sparse.shape[1])
                    thr = (
                        torch.topk(h_sparse, k_relaxed, dim=1)
                        .values[:, -1]
                        .unsqueeze(1)
                    )
                    h_sparse = torch.where(
                        h_sparse >= thr, h_sparse, torch.zeros_like(h_sparse)
                    )

                    # PO BOOST - výpočet scores:
                    recon_z_after = self.sae.dec(h_sparse)
                    recon_z_norm_after = torch.nn.functional.normalize(
                        recon_z_after, dim=-1
                    )
                    scores_after = torch.matmul(
                        recon_z_norm_after, A_norm.T
                    ) - user_vector.unsqueeze(0)
                    scores_after_np = scores_after.squeeze().numpy()

                    # ANALÝZA BOOST EFEKTU:
                    score_changes = scores_after_np - scores_before_np

                    # Top 10 nejvíce boostnutých filmů:
                    most_boosted_indices = np.argsort(-score_changes)[:10]
                    logger.info(
                        f"most_boosted_indices: {most_boosted_indices}, type {type(most_boosted_indices)}, len {len(most_boosted_indices)} "
                    )
                    logger.info("TOP 10 MOST BOOSTED MOVIES:")
                    for i, idx in enumerate(most_boosted_indices):
                        before_score = scores_before_np[idx]
                        after_score = scores_after_np[idx]
                        boost_change = score_changes[idx]
                        title = f"Item {idx}"
                        if (
                            hasattr(self.loader, "items_df")
                            and hasattr(self.loader.items_df, "iloc")
                            and len(self.loader.items_df) > idx
                            and idx >= 0
                        ):

                            movie_info = self.loader.items_df.iloc[idx]
                            if "title" in movie_info:
                                raw_title = str(movie_info["title"])
                                # Zkrátit a očistit title:
                                title = (
                                    raw_title[:40]
                                    .encode("ascii", "ignore")
                                    .decode("ascii")
                                )
                                if (
                                    not title.strip()
                                ):  # Pokud title je prázdný po čištění
                                    title = f"Item {idx}"
                            else:
                                title = f"Item {idx} (no title)"
                        else:
                            title = f"Item {idx} (out of bounds)"
                        # Log výsledek:
                        logger.info(f"Processed title {title}")
                        logger.info(
                            f"  {i+1}. {title} before:{before_score:.4f} after:{after_score:.4f} change:{boost_change:+.4f}"
                        )

                    boost_effect = torch.abs(h_sparse - h_sparse_original).mean().item()
                    logger.info(
                        f"Query boost effect (mean absolute change): {boost_effect:.6f}"
                    )
                    logger.info(
                        f"Overall score change: mean={score_changes.mean():.6f}, max_boost={score_changes.max():.6f}, max_penalty={score_changes.min():.6f}"
                    )

                    boosted_stats = {
                        "mean": h_sparse.mean().item(),
                        "std": h_sparse.std().item(),
                        "max": h_sparse.max().item(),
                        "nonzero_count": (h_sparse != 0).sum().item(),
                    }
                    logger.info(f"Boosted sparse representation stats: {boosted_stats}")
                else:
                    logger.info("Query boost vector is None")
            else:
                logger.info("No query provided - no boost applied")

            # Pokračuj s konečným výpočtem:
            recon_z = self.sae.dec(h_sparse)
            A_norm = torch.nn.functional.normalize(self.elsa.A, dim=-1)
            recon_z_norm = torch.nn.functional.normalize(recon_z, dim=-1)
            scores = torch.matmul(recon_z_norm, A_norm.T) - user_vector.unsqueeze(0)

        scores_np = scores.squeeze().numpy()

        # Log score statistics before filtering
        logger.info(
            f"Recommendation scores stats: mean={scores_np.mean():.4f}, "
            f"std={scores_np.std():.4f}, max={scores_np.max():.4f}, min={scores_np.min():.4f}"
        )

        # PŘIDEJ ROZŠÍŘENÉ LOGOVÁNÍ:
        logger.info("DETAILED SCORE ANALYSIS:")

        # Top 10 scores před filtrováním:
        top_unfiltered_indices = np.argsort(-scores_np)[:10]
        logger.info("Top 10 scores (before filtering):")
        for i, idx in enumerate(top_unfiltered_indices):
            score = scores_np[idx]
            try:
                if hasattr(self.loader, "items_df") and idx < len(self.loader.items_df):
                    movie_info = self.loader.items_df.iloc[idx]
                    title = str(movie_info.get("title", f"Item {idx}"))[
                        :50
                    ]  # Limit title length
                    title = title.encode("ascii", "ignore").decode("ascii")
                    logger.info(
                        f"  {i+1:2d}. {title:<30} (Item {idx:4d}, score: {score:.4f})"
                    )
                else:
                    logger.info(f"  {i+1:2d}. Item {idx:4d} (score: {score:.4f})")
            except Exception as e:
                logger.info(
                    f"  {i+1:2d}. Item {idx:4d} (score: {score:.4f}) [error: {str(e)[:20]}]"
                )

        # Analýza score distribuce:
        positive_scores = scores_np[scores_np > 0]
        negative_scores = scores_np[scores_np < 0]

        logger.info(f"Score distribution:")
        logger.info(
            f"  Positive scores: {len(positive_scores)} items (mean: {positive_scores.mean():.4f})"
            if len(positive_scores) > 0
            else "  No positive scores!"
        )
        logger.info(
            f"  Negative scores: {len(negative_scores)} items (mean: {negative_scores.mean():.4f})"
            if len(negative_scores) > 0
            else "  No negative scores"
        )

        mask = np.isin(np.arange(self.num_items), filter_out_items)
        scores_np[mask] = -np.inf
        top_indices = np.argsort(-scores_np)[:k]

        logger.info(f"Top {min(5, k)} recommendations (after filtering):")
        # ...rest stays same...
        mask = np.isin(np.arange(self.num_items), filter_out_items)
        scores_np[mask] = -np.inf
        top_indices = np.argsort(-scores_np)[:k]

        logger.info(f"Top {min(5, k)} recommendations:")
        for i, idx in enumerate(top_indices[:5]):
            score = scores.squeeze()[idx].item()
            try:
                if hasattr(self.loader, "items_df") and idx < len(self.loader.items_df):
                    movie_info = self.loader.items_df.iloc[idx]
                    title = str(movie_info.get("title", f"Item {idx}"))
                    title = title.encode("ascii", "ignore").decode("ascii")
                    logger.info(f"  {i+1}. {title} (Item {idx}, score: {score:.4f})")
                else:
                    logger.info(f"  {i+1}. Item {idx} (score: {score:.4f})")
            except Exception as e:
                logger.info(
                    f"  {i+1}. Item {idx} (score: {score:.4f}) [title error: {str(e)}]"
                )

        return top_indices.tolist()
