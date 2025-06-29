import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset

# Načti sparse train data
with open("processed_train.pkl", "rb") as f:
    X_train = pickle.load(f)

num_users, num_items = X_train.shape
latent_dim = 512  # embedding dim

print(f"Train dataset: {num_users} users × {num_items} items")


# Dataset, který dávkuje řádky jako dense tensor
class SparseDataset(Dataset):
    def __init__(self, csr_matrix):
        self.data = csr_matrix

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32)


dataset = SparseDataset(X_train)


# Model
class ELSA(nn.Module):
    def __init__(self, num_items, latent_dim):
        super(ELSA, self).__init__()
        self.A = nn.Parameter(torch.randn(num_items, latent_dim))

    def forward(self, x):
        # Encoder
        z = torch.matmul(x, self.A)
        # Decoder
        recon = torch.matmul(z, self.A.T) - x
        return recon


model = ELSA(num_items, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()


# Normalizace řádků matice A
def normalize_A(A):
    with torch.no_grad():
        norms = A.norm(p=2, dim=1, keepdim=True)
        A /= norms


# Trénovací smyčka
n_epochs = 50
batch_size = 256

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for x_batch in loader:
        optimizer.zero_grad()
        recon = model(x_batch)
        l2_lambda = 1e-4
        l2_reg = l2_lambda * torch.norm(model.A, p=2)
        loss = criterion(recon, x_batch) + l2_reg
        loss.backward()
        optimizer.step()
        normalize_A(model.A)
        epoch_loss += loss.item() * x_batch.size(0)
    avg_loss = epoch_loss / num_users
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

# Ulož model
torch.save(model.state_dict(), "elsa_model.pt")
