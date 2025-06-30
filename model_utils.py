# model_utils.py
import torch
import pickle

from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k

hidden_dim = 4096


def load_models_and_data():
    # Načti item2index
    with open("data/item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    index2item = {v: k for k, v in item2index.items()}
    num_items = len(item2index)

    # Načti ELSA model
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load("models/elsa_model.pt"))
    elsa.eval()

    # Načti SAE model
    sae = TopKSAE(latent_dim, hidden_dim, k)
    sae.load_state_dict(torch.load("models/sae_model.pt"))
    sae.eval()

    # Načti tag mapu
    tag_map = torch.load("models/tag_neuron_map.pt")
    tag_tensor = tag_map["tag_tensor"]
    unique_tags = tag_map["unique_tags"]

    # Načti trénovací data
    with open("data/processed_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    X_train = X_train.toarray()

    return elsa, sae, item2index, index2item, tag_tensor, unique_tags, X_train
