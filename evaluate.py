import torch
import pickle
import numpy as np

from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k

hidden_dim = 1024

# Load item2index
with open("data/item2index.pkl", "rb") as f:
    item2index = pickle.load(f)
num_items = len(item2index)

# Load models
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("models/elsa_model.pt"))
elsa.eval()

sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("models/sae_model.pt"))
sae.eval()

# Load train and test matrices
with open("data/processed_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("data/processed_test.pkl", "rb") as f:
    X_test = pickle.load(f)

X_train = X_train.toarray()
X_test = X_test.toarray()


# Metric functions
def recall_at_k(y_true, y_pred, k):
    hits = y_true[y_pred[:k]].sum()
    total_relevant = y_true.sum()
    return hits / total_relevant if total_relevant > 0 else np.nan


def ndcg_at_k(y_true, y_pred, k):
    hits = y_true[y_pred[:k]]
    if hits.sum() == 0:
        return 0.0
    gains = hits / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    ideal_hits = np.sort(y_true)[::-1][:k]
    ideal_gains = ideal_hits / np.log2(np.arange(2, k + 2))
    idcg = ideal_gains.sum()
    return dcg / idcg if idcg > 0 else 0.0


# Lists to store results
recall_20_elsa = []
recall_50_elsa = []
ndcg_20_elsa = []
ndcg_50_elsa = []

recall_20_sae = []
recall_50_sae = []
ndcg_20_sae = []
ndcg_50_sae = []

# Evaluate
for i in range(X_test.shape[0]):
    user_train = torch.tensor(X_train[i], dtype=torch.float32).unsqueeze(0)
    user_test = X_test[i]

    if user_test.sum() == 0:
        continue

    with torch.no_grad():
        # ELSA-only predictions
        z_elsa = torch.matmul(user_train, elsa.A)
        recon_elsa = torch.matmul(z_elsa, elsa.A.T)
        scores_elsa = recon_elsa - user_train

        scores_elsa_np = scores_elsa.squeeze().numpy()
        scores_elsa_np[X_train[i] > 0] = -np.inf
        top_indices_elsa = np.argsort(-scores_elsa_np)

        # ELSA+SAE predictions
        h = torch.relu(sae.enc(z_elsa))
        h_sparse = torch.where(
            h >= torch.topk(h, k, dim=1).values[:, -1].unsqueeze(1),
            h,
            torch.zeros_like(h),
        )
        recon_sae = sae.dec(h_sparse)
        scores_sae = torch.matmul(recon_sae, elsa.A.T) - user_train

        scores_sae_np = scores_sae.squeeze().numpy()
        scores_sae_np[X_train[i] > 0] = -np.inf
        top_indices_sae = np.argsort(-scores_sae_np)

    # ELSA metrics
    recall_20_elsa.append(recall_at_k(user_test, top_indices_elsa, 20))
    recall_50_elsa.append(recall_at_k(user_test, top_indices_elsa, 50))
    ndcg_20_elsa.append(ndcg_at_k(user_test, top_indices_elsa, 20))
    ndcg_50_elsa.append(ndcg_at_k(user_test, top_indices_elsa, 50))

    # SAE metrics
    recall_20_sae.append(recall_at_k(user_test, top_indices_sae, 20))
    recall_50_sae.append(recall_at_k(user_test, top_indices_sae, 50))
    ndcg_20_sae.append(ndcg_at_k(user_test, top_indices_sae, 20))
    ndcg_50_sae.append(ndcg_at_k(user_test, top_indices_sae, 50))

print(f"Evaluated {len(recall_20_elsa)} users with test interactions.\n")

print("==== ELSA Only ====")
print(f"Mean Recall@20: {np.nanmean(recall_20_elsa):.4f}")
print(f"Mean Recall@50: {np.nanmean(recall_50_elsa):.4f}")
print(f"Mean NDCG@20:   {np.nanmean(ndcg_20_elsa):.4f}")
print(f"Mean NDCG@50:   {np.nanmean(ndcg_50_elsa):.4f}")

print("\n==== ELSA + SAE ====")
print(f"Mean Recall@20: {np.nanmean(recall_20_sae):.4f}")
print(f"Mean Recall@50: {np.nanmean(recall_50_sae):.4f}")
print(f"Mean NDCG@20:   {np.nanmean(ndcg_20_sae):.4f}")
print(f"Mean NDCG@50:   {np.nanmean(ndcg_50_sae):.4f}")
