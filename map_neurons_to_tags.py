import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
import pickle

# Load CFAE and SAE models
from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k

# Load metadata
tags_df = pd.read_csv("data/tags.csv")
with open("data/item2index.pkl", "rb") as f:
    item2index = pickle.load(f)

index2item = {v: k for k, v in item2index.items()}
num_items = len(item2index)
hidden_dim = 1024

# Convert tags to lowercase
tags_df["tag"] = tags_df["tag"].str.lower()

# Map items to their tag lists
item_tags = defaultdict(list)
for _, row in tags_df.iterrows():
    iid = row["movieId"]
    if iid in item2index:
        item_tags[item2index[iid]].append(row["tag"])

# Load CFAE model
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("models/elsa_model.pt"))
elsa.eval()

# Load SAE model
sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("models/sae_model.pt"))
sae.eval()

# Prepare CFAE item embeddings as the direct rows of A
with torch.no_grad():
    embeddings = elsa.A.clone()

print("Item embeddings loaded.")

# SAE encoding (batched)
batch_size = 1024
h_list = []

print("Creating sparse item embeddings...")
with torch.no_grad():
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_emb = embeddings[start:end]
        h_batch = torch.relu(sae.enc(batch_emb))
        h_sparse_batch = torch.where(
            h_batch >= torch.topk(h_batch, k, dim=1).values[:, -1].unsqueeze(1),
            h_batch,
            torch.zeros_like(h_batch),
        )
        h_list.append(h_sparse_batch)

h_sparse = torch.cat(h_list, dim=0)

print("Sparse embeddings ready.")

# Build neuron × tag table (sum of activations)
tag_counts = defaultdict(lambda: torch.zeros(hidden_dim))

print("Aggregating activations by tag...")
for idx in range(num_items):
    tags = item_tags.get(idx, [])
    for tag in tags:
        tag_counts[tag] += h_sparse[idx]

# List of unique tags
unique_tags = list(tag_counts.keys())

# Convert to tensor (tags × neurons)
tag_tensor = torch.stack([tag_counts[tag] for tag in unique_tags])

# Normalization
norms = torch.norm(tag_tensor, p=2, dim=1, keepdim=True)
tag_tensor = tag_tensor / norms

# Save the map
torch.save(
    {"unique_tags": unique_tags, "tag_tensor": tag_tensor}, "models/tag_neuron_map.pt"
)

print(f"Done. Map contains {len(unique_tags)} tags.")
