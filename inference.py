import torch
import numpy as np
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Import models and constants
from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k

# Hidden dimension of the SAE; must match training
hidden_dim = 1024

# Load item2index mapping
with open("data/item2index.pkl", "rb") as f:
    item2index = pickle.load(f)
index2item = {v: k for k, v in item2index.items()}
num_items = len(item2index)

# Load tag neuron map
tag_map = torch.load("models/tag_neuron_map.pt")
tag_tensor = tag_map["tag_tensor"]

# Load precomputed tag embeddings
embedding_data = torch.load("models/tag_embeddings.pt")
unique_tags = embedding_data["unique_tags"]
tag_embeddings = embedding_data["embeddings"]

# Load sentence-transformer model
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load CFAE model
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("models/elsa_model.pt"))
elsa.eval()

# Load SAE model
sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("models/sae_model.pt"))
sae.eval()

# Load training matrix for simulating interactions
with open("data/processed_train.pkl", "rb") as f:
    X_train = pickle.load(f)
X_train = X_train.toarray()

# Load movie metadata
movies_df = pd.read_csv("data/movies.csv")
movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))


# Function to map text query to neurons
def query_to_neurons(query, top_n=5):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, tag_embeddings)[0]

# Top-N indices and their scores
    top_results = torch.topk(cos_scores, k=top_n)
    indices = top_results.indices
    scores = top_results.values

    # Normalize weights to sum to 1
    total = scores.sum()
    if total > 0:
        weights = scores / total
    else:
        weights = torch.ones_like(scores) / len(scores)

    print("Matched tags with weights:")
    for i, w in zip(indices, weights):
        print(f"- {unique_tags[i]} (weight={w:.3f})")

    # Weighted average of activations
    weighted_sum = torch.zeros(tag_tensor.shape[1], dtype=tag_tensor.dtype)
    for idx, weight in zip(indices, weights):
        weighted_sum += weight * tag_tensor[idx]

    return weighted_sum


# Function to generate recommendations
def recommend(user_idx, boost_vector=None, alpha=0.85, topk=10):
    user_vector = torch.tensor(X_train[user_idx], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        z = torch.matmul(user_vector, elsa.A)
        h = torch.relu(sae.enc(z))
        h_sparse = torch.where(
            h >= torch.topk(h, k, dim=1).values[:, -1].unsqueeze(1),
            h,
            torch.zeros_like(h),
        )
        if boost_vector is not None:
            h_sparse = alpha * h_sparse + (1 - alpha) * boost_vector.unsqueeze(0)
        recon_z = sae.dec(h_sparse)
        scores = torch.matmul(recon_z, elsa.A.T) - user_vector

    scores_np = scores.squeeze().numpy()
    scores_np[X_train[user_idx] > 0] = -np.inf
    top_indices = np.argsort(-scores_np)[:topk]
    return top_indices, scores_np[top_indices]


# Main
if __name__ == "__main__":
    print("==== Inference pipeline ====")
    user_idx = int(input(f"Enter user index (0 to {X_train.shape[0]-1}): "))
    query = input("Enter your query in English (e.g., 'I want a dark thriller'): ")

    boost = query_to_neurons(query)
    if boost is None:
        print("No boosting vector found, generating only baseline recommendations.")
        boost = None

    print("\n=== Generating recommendations... ===")
    recs_no_boost, scores_no_boost = recommend(user_idx, boost_vector=None)
    recs_normal_boost, scores_normal_boost = recommend(
        user_idx, boost_vector=boost, alpha=0.7
    )
    recs_strong_boost, scores_strong_boost = recommend(
        user_idx, boost_vector=boost, alpha=0.2
    )

    def print_recommendations(title, indices, scores):
        print(f"\n--- {title} ---")
        for idx, score in zip(indices, scores):
            movie_id = index2item[idx]
            title = movieid_to_title.get(movie_id, "Unknown title")
            print(f"{title} (MovieID {movie_id}), score {score:.4f}")

    print_recommendations("1) No Boosting", recs_no_boost, scores_no_boost)
    print_recommendations(
        "2) Normal Boosting (alpha=0.7)", recs_normal_boost, scores_normal_boost
    )
    print_recommendations(
        "3) Strong Boosting (alpha=0.2)", recs_strong_boost, scores_strong_boost
    )
