import torch
import torch.nn as nn
import pickle
import numpy as np

# Načti modely a mappingy
from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k

hidden_dim = 4096
batch_size = 512

# Načti item2index
with open("item2index.pkl", "rb") as f:
    item2index = pickle.load(f)
index2item = {v: k for k, v in item2index.items()}
num_items = len(item2index)

# Načti mapu tagů
tag_map = torch.load("tag_neuron_map.pt")
tag_tensor = tag_map["tag_tensor"]
unique_tags = tag_map["unique_tags"]

# Načti modely
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("elsa_model.pt"))
elsa.eval()

sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("sae_model.pt"))
sae.eval()

# Načti trénovací matici pro simulaci interakcí
with open("processed_train.pkl", "rb") as f:
    X_train = pickle.load(f)
X_train = X_train.toarray()


# Funkce pro mapping plain-text dotazu na neurony
def query_to_neurons(query, top_n=3):
    query = query.lower()
    matches = [i for i, tag in enumerate(unique_tags) if query in tag]
    if not matches:
        print("Nenalezen žádný odpovídající tag.")
        return None
    selected = matches[:top_n]
    print(f"Nalezené tagy: {[unique_tags[i] for i in selected]}")
    # Průměr aktivací odpovídajících tagů
    return tag_tensor[selected].mean(dim=0)


# Funkce pro inference
def recommend(user_idx, boost_vector=None, alpha=0.85, topk=10):
    user_vector = torch.tensor(X_train[user_idx], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        # CFAE embedding
        z = torch.matmul(user_vector, elsa.A)
        # SAE encoding
        h = torch.relu(sae.enc(z))
        h_sparse = torch.where(
            h >= torch.topk(h, k, dim=1).values[:, -1].unsqueeze(1),
            h,
            torch.zeros_like(h),
        )
        # Boosting
        if boost_vector is not None:
            h_sparse = alpha * h_sparse + (1 - alpha) * boost_vector.unsqueeze(0)
        # Rekonstrukce
        recon_z = sae.dec(h_sparse)
        scores = torch.matmul(recon_z, elsa.A.T) - user_vector

    scores_np = scores.squeeze().numpy()
    scores_np[X_train[user_idx] > 0] = -np.inf
    top_indices = np.argsort(-scores_np)[:topk]
    return top_indices, scores_np[top_indices]


# Hlavní
if __name__ == "__main__":
    print("==== Inference pipeline ====")
    user_idx = int(input(f"Zadej index uživatele (0 až {X_train.shape[0]-1}): "))
    query = input("Zadej dotaz (část tagu, např. 'thriller'): ")

    boost = query_to_neurons(query)
    if boost is None:
        print("Generuji bez boostingu...")
        boost = None

    recs, rec_scores = recommend(user_idx, boost_vector=boost)

    print("\n=== Doporučení ===")
    for idx, score in zip(recs, rec_scores):
        movie_id = index2item[idx]
        print(f"Item index {idx} (MovieID {movie_id}), skóre {score:.4f}")
