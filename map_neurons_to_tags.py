import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
import pickle

# Načti CFAE a SAE model
from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k

# Načti metadata
tags_df = pd.read_csv("data/tags.csv")
with open("item2index.pkl", "rb") as f:
    item2index = pickle.load(f)

index2item = {v: k for k, v in item2index.items()}
num_items = len(item2index)
hidden_dim = 4096

# Převeď tagy na lower case
tags_df["tag"] = tags_df["tag"].str.lower()

# Mapuj item -> seznam tagů
item_tags = defaultdict(list)
for _, row in tags_df.iterrows():
    iid = row["movieId"]
    if iid in item2index:
        item_tags[item2index[iid]].append(row["tag"])

# Načti CFAE model
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("elsa_model.pt"))
elsa.eval()

# Načti SAE model
sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("sae_model.pt"))
sae.eval()

# Připrav CFAE embeddingy položek jako jen přímé řádky A
with torch.no_grad():
    embeddings = elsa.A.clone()

print("Embeddingy položek načteny.")

# SAE encoding (po dávkách)
batch_size = 1024
h_list = []

print("Vytvářím sparse embeddingy položek...")
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

print("Sparse embeddingy hotové.")

# Vytvoř tabulku neuron × tag (součet aktivací)
tag_counts = defaultdict(lambda: torch.zeros(hidden_dim))

print("Agreguji aktivace podle tagů...")
for idx in range(num_items):
    tags = item_tags.get(idx, [])
    for tag in tags:
        tag_counts[tag] += h_sparse[idx]

# Seznam unikátních tagů
unique_tags = list(tag_counts.keys())

# Převod na tensor (tags × neurons)
tag_tensor = torch.stack([tag_counts[tag] for tag in unique_tags])

# Normalizace
norms = torch.norm(tag_tensor, p=2, dim=1, keepdim=True)
tag_tensor = tag_tensor / norms

# Ulož mapu
torch.save({"unique_tags": unique_tags, "tag_tensor": tag_tensor}, "tag_neuron_map.pt")

print(f"Done. Map contains {len(unique_tags)} tags.")
