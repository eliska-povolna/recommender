import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Načti trénovací embeddingy uživatelů
with open("processed_train.pkl", "rb") as f:
    X_train = pickle.load(f)
X_train = X_train.toarray()
X_tensor = torch.tensor(X_train, dtype=torch.float32)

# Načti CFAE model (ELSA)
from train_elsa import ELSA

latent_dim = 512
hidden_dim = 4096
k = 16

# CFAE model pro generování embeddingů
num_items = X_tensor.shape[1]
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("elsa_model.pt"))
elsa.eval()

with torch.no_grad():
    embeddings = torch.matmul(X_tensor, elsa.A)

print("Embedding shape:", embeddings.shape)  # (num_users, latent_dim)


# TopK aktivace
def topk_activation(x, k):
    """
    Nuluje všechny hodnoty kromě k největších.
    """
    values, indices = torch.topk(x, k)
    mask = torch.zeros_like(x)
    mask.scatter_(1, indices, 1.0)
    return x * mask


# Sparse Autoencoder
class TopKSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super(TopKSAE, self).__init__()
        self.k = k
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = torch.relu(self.enc(x))
        h_sparse = topk_activation(h, self.k)
        out = self.dec(h_sparse)
        return out, h_sparse


sae = TopKSAE(input_dim=latent_dim, hidden_dim=hidden_dim, k=k)

optimizer = optim.Adam(sae.parameters(), lr=5e-4)
criterion = nn.MSELoss()

# Trénovací smyčka
n_epochs = 100
batch_size = 256

dataset = torch.utils.data.TensorDataset(embeddings)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch in loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        recon, h_sparse = sae(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    avg_loss = epoch_loss / embeddings.shape[0]
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

# Ulož model
torch.save(sae.state_dict(), "sae_model.pt")

# Ulož sparse embeddingy všech uživatelů
with torch.no_grad():
    h = torch.relu(sae.enc(embeddings))
    h_sparse = topk_activation(h, k)

torch.save(h_sparse, "sparse_embeddings.pt")
print("SAE model trained.")
