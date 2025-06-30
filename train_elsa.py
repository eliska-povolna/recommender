import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

latent_dim = 128  # embedding dimension


# Model matching the reference implementation
class ELSA(nn.Module):
    def __init__(self, num_items, latent_dim):
        super(ELSA, self).__init__()
        self.A = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty([num_items, latent_dim]))
        )

    def forward(self, x):
        A = torch.nn.functional.normalize(self.A, dim=-1)
        z = torch.matmul(x, A)
        recon = torch.matmul(z, A.T)
        return recon


# Normalized MSE loss
class NMSELoss(nn.Module):
    def forward(self, input, target):
        return torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(input, dim=-1),
            torch.nn.functional.normalize(target, dim=-1),
            reduction="mean",
        )


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
    # Load sparse training data
    with open("data/processed_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    num_users, num_items = X_train.shape
    print(f"Train dataset: {num_users} users x {num_items} items")

    # Split users into train/validation
    user_indices = np.arange(num_users)
    train_user_indices, val_user_indices = train_test_split(
        user_indices, test_size=0.1, random_state=42
    )

    X_train_data = X_train[train_user_indices]
    X_val_data = X_train[val_user_indices]

    # Dataset yielding dense rows
    class SparseDataset(Dataset):
        def __init__(self, csr_matrix):
            self.data = csr_matrix

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            row = self.data[idx].toarray().squeeze()
            return torch.tensor(row, dtype=torch.float32)

    train_dataset = SparseDataset(X_train_data)
    val_dataset = SparseDataset(X_val_data)

    model = ELSA(num_items, latent_dim)
    optimizer = optim.NAdam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = NMSELoss()

    n_epochs = 100
    batch_size = 256
    patience = 10  # early stopping patience

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch in train_loader:
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        avg_train_loss = epoch_loss / len(train_dataset)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch in val_loader:
                recon = model(x_batch)
                batch_loss = criterion(recon, x_batch)
                val_loss += batch_loss.item() * x_batch.size(0)
        avg_val_loss = val_loss / len(val_dataset)

        print(
            f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/elsa_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(torch.load("models/elsa_model.pt"))
    torch.save(model.state_dict(), "models/elsa_model.pt")
    print("Best model saved to elsa_model.pt")

    # Evaluate Recall@20 and NDCG@20 on validation users
    recall_scores = []
    ndcg_scores = []
    model.eval()

    with torch.no_grad():
        for i in range(X_val_data.shape[0]):
            user_train = X_train_data[i].toarray().squeeze()
            user_val = X_val_data[i].toarray().squeeze()
            user_vector = torch.tensor(user_train, dtype=torch.float32).unsqueeze(0)
            recon = model(user_vector)
            scores = recon - user_vector

            scores_np = scores.squeeze().numpy()
            # Mask training items only
            scores_np[user_train > 0] = -np.inf
            top_indices = np.argsort(-scores_np)

            y_true = user_val  # Only held-out items
            recall = recall_at_k(y_true, top_indices, 20)
            ndcg = ndcg_at_k(y_true, top_indices, 20)

            recall_scores.append(recall)
            ndcg_scores.append(ndcg)

    print("\nValidation Recall@20 and NDCG@20 on held-out users:")
    print(f"Mean Recall@20: {np.nanmean(recall_scores):.4f}")
    print(f"Mean NDCG@20:   {np.nanmean(ndcg_scores):.4f}")
