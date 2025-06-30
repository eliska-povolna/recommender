import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv("data/ratings.csv")

# Convert to implicit feedback (1 if rating >=4)
df["implicit"] = (df["rating"] >= 4).astype(np.float32)

# Remove interactions with implicit == 0
df = df[df["implicit"] > 0]

# Remap userId and movieId to indices (0,1,2,...)
user_ids = df["userId"].unique()
item_ids = df["movieId"].unique()

user2index = {uid: idx for idx, uid in enumerate(user_ids)}
item2index = {iid: idx for idx, iid in enumerate(item_ids)}

df["user_idx"] = df["userId"].map(user2index)
df["item_idx"] = df["movieId"].map(item2index)

num_users = len(user2index)
num_items = len(item2index)

# Create sparse matrix
X = csr_matrix(
    (np.ones(len(df)), (df["user_idx"], df["item_idx"])), shape=(num_users, num_items)
)

# Drop users with <5 interactions
user_activity = np.array(X.sum(axis=1)).flatten()
active_users = np.where(user_activity >= 5)[0]
X = X[active_users]

# Split train/test (disjoint user sets)
all_users = np.arange(X.shape[0])
train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)

X_train = X[train_users]
X_test = X[test_users]

# Save results
with open("processed_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("processed_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

# Save item2index for later loading
with open("item2index.pkl", "wb") as f:
    pickle.dump(item2index, f)

print(
    f"Preprocessing done. Train users: {X_train.shape[0]}, Test users: {X_test.shape[0]}, Items: {X_train.shape[1]}"
)
