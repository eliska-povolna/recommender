import torch
import pickle
import numpy as np

from train_elsa import ELSA, latent_dim

from train_sae import TopKSAE, k, hidden_dim

# Load item2index
with open("data/item2index.pkl", "rb") as f:
    item2index = pickle.load(f)
num_items = len(item2index)

# Load models
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("models/elsa_model_best.pt"))
elsa.eval()

sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("models/sae_model_r4_k32.pt"))
sae.eval()

# Load test matrices
with open("data/processed_test.pkl", "rb") as f:
    X_test = pickle.load(f)


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

for i in range(X_test.shape[0]):
    user_full = X_test.getrow(i).toarray().squeeze()

    # Skip users with insufficient interactions
    nonzero_items = np.where(user_full > 0)[0]
    if len(nonzero_items) < 5:
        continue

    # 20% item holdout per user
    np.random.seed(42 + i)
    n_holdout = max(1, int(len(nonzero_items) * 0.2))
    holdout_items = np.random.choice(nonzero_items, size=n_holdout, replace=False)

    user_input = user_full.copy()
    user_target = np.zeros_like(user_full)

    user_input[holdout_items] = 0
    user_target[holdout_items] = user_full[holdout_items]

    user_train = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # ELSA-only predictions
        z_elsa = torch.matmul(user_train, elsa.A)
        z_elsa_norm = torch.nn.functional.normalize(z_elsa, dim=-1)
        recon_elsa = torch.matmul(z_elsa_norm, elsa.A.T)
        scores_elsa = recon_elsa - user_train

        scores_elsa_np = scores_elsa.squeeze().numpy()
        scores_elsa_np[user_input > 0] = -np.inf
        top_indices_elsa = np.argsort(-scores_elsa_np)

        # ELSA+SAE predictions
        recon_z, h_sparse, h_pre = sae(z_elsa_norm)

        recon_sae = torch.matmul(recon_z, elsa.A.T)
        scores_sae = recon_sae - user_train

        scores_sae_np = scores_sae.squeeze().numpy()
        scores_sae_np[user_input > 0] = -np.inf
        top_indices_sae = np.argsort(-scores_sae_np)

    # ELSA metrics
    recall_20_elsa.append(recall_at_k(user_target, top_indices_elsa, 20))
    recall_50_elsa.append(recall_at_k(user_target, top_indices_elsa, 50))
    ndcg_20_elsa.append(ndcg_at_k(user_target, top_indices_elsa, 20))
    ndcg_50_elsa.append(ndcg_at_k(user_target, top_indices_elsa, 50))

    # SAE metrics
    recall_20_sae.append(recall_at_k(user_target, top_indices_sae, 20))
    recall_50_sae.append(recall_at_k(user_target, top_indices_sae, 50))
    ndcg_20_sae.append(ndcg_at_k(user_target, top_indices_sae, 20))
    ndcg_50_sae.append(ndcg_at_k(user_target, top_indices_sae, 50))

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


import csv, os, subprocess, datetime as dt

metrics = {
    "ts": dt.datetime.now().isoformat(timespec="seconds"),
    "commit": subprocess.getoutput("git rev-parse --short HEAD") or "n/a",
    "users_eval": len(recall_20_elsa),
    "recall20_elsa": float(np.nanmean(recall_20_elsa)),
    "recall50_elsa": float(np.nanmean(recall_50_elsa)),
    "ndcg20_elsa": float(np.nanmean(ndcg_20_elsa)),
    "ndcg50_elsa": float(np.nanmean(ndcg_50_elsa)),
    "recall20_sae": float(np.nanmean(recall_20_sae)),
    "recall50_sae": float(np.nanmean(recall_50_sae)),
    "ndcg20_sae": float(np.nanmean(ndcg_20_sae)),
    "ndcg50_sae": float(np.nanmean(ndcg_50_sae)),
}

path = "C:\\Users\\elisk\\Desktop\\2024-25\\LS\\Recommenders\\logs\\metrics.csv"
write_header = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
    if write_header:
        w.writeheader()
    w.writerow(metrics)
print("Logged metrics to", path)
