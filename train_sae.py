import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from train_elsa import ELSA, latent_dim

hidden_dim = 1024
k = 32  # sparsity level


# TopK activation function
def topk_activation(x, k):
    """
    Zero out all values except top-k largest.
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


# Recall@k
def recall_at_k(y_true, y_pred, k):
    hits = y_true[y_pred[:k]].sum()
    total_relevant = y_true.sum()
    return hits / total_relevant if total_relevant > 0 else np.nan


# NDCG@k
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


if __name__ == "__main__":
    # Load training data
    with open("data/processed_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    X_train = X_train.toarray()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    # Load trained ELSA model
    num_items = X_tensor.shape[1]
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load("models/elsa_model.pt"))
    elsa.eval()

    # Generate and normalize embeddings
    with torch.no_grad():
        embeddings = torch.matmul(X_tensor, elsa.A)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    print("Embedding shape:", embeddings.shape)  # (num_users, latent_dim)

    # Split users into train/validation
    num_users = embeddings.shape[0]
    user_indices = np.arange(num_users)
    train_indices, val_indices = train_test_split(
        user_indices, test_size=0.1, random_state=42
    )

    train_tensor = torch.tensor(embeddings[train_indices], dtype=torch.float32)
    val_tensor = torch.tensor(embeddings[val_indices], dtype=torch.float32)
    val_input_sparse = X_train[val_indices]

    # Initialize SAE
    sae = TopKSAE(input_dim=latent_dim, hidden_dim=hidden_dim, k=k)
    optimizer = optim.Adam(sae.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.CosineEmbeddingLoss()

    n_epochs = 200
    batch_size = 1024
    patience = 10

    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        sae.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            recon, h_sparse = sae(x_batch)
            labels = torch.ones(x_batch.size(0))
            loss = criterion(recon, x_batch, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)
        avg_train_loss = epoch_train_loss / len(train_tensor)

        # Validation loss
        sae.eval()
        with torch.no_grad():
            recon_val, _ = sae(val_tensor)
            labels_val = torch.ones(val_tensor.size(0))
            val_loss = criterion(recon_val, val_tensor, labels_val).item()

        print(
            f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(sae.state_dict(), "models/sae_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    # Load best model
    sae.load_state_dict(torch.load("models/sae_model.pt"))

    # Save sparse embeddings of all users
    with torch.no_grad():
        h_all = torch.relu(sae.enc(embeddings))
        h_sparse_all = topk_activation(h_all, k)

    torch.save(h_sparse_all, "models/sparse_embeddings.pt")
    print("Best SAE model and embeddings saved.")
