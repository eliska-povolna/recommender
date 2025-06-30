import torch
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# For example, load unique_tags from tag_neuron_map.pt
tag_map = torch.load("models/tag_neuron_map.pt")
unique_tags = tag_map["unique_tags"]

tag_embeddings = embed_model.encode(unique_tags, convert_to_tensor=True)

torch.save(
    {"unique_tags": unique_tags, "embeddings": tag_embeddings},
    "models/tag_embeddings.pt",
)
