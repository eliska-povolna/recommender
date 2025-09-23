import torch
from sentence_transformers import SentenceTransformer, util
import os
import glob


def preprocess_labels_for_embedding(unique_labels):
    """Process both genres and tags for embedding, keeping context prefixes"""
    processed_labels = []

    for label in unique_labels:
        if label.startswith("genre:"):
            clean_label = label[6:]  # Remove "genre:" prefix
            processed_label = f"{clean_label} movie genre"
        elif label.startswith("tag:"):
            clean_label = label[4:]  # Remove "tag:" prefix
            processed_label = f"{clean_label} movie tag"
        else:
            processed_label = label.lower().strip()

        # Handle compound tags (hyphens to spaces)
        if "-" in processed_label:
            processed_label = processed_label.replace("-", " ")

        processed_labels.append(processed_label)

    return processed_labels


print("Loading sentence transformer model...")
embed_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
print(f"Loaded model: all-distilroberta-v1")

# Find all tag neuron mapping files
tag_files = glob.glob("models/tag_neuron_map_*.pt")
if not tag_files:
    print("❌ No tag neuron mapping files found!")
    print("Make sure you've run map_neurons_to_tags_final.py first")
    exit(1)

print(f"Found {len(tag_files)} method files:")
for file in tag_files:
    print(f"  - {os.path.basename(file)}")

# Load summary to get best method info
try:
    summary = torch.load("models/tag_neuron_summary.pt", weights_only=False)
    best_method = summary.get("best_method", "tf_idf")
    print(f"Best method: {best_method}")
except:
    print("⚠️ No summary file found, will process all methods")
    best_method = None

# Process each method file
all_embeddings = {}

for file_path in tag_files:
    method_key = (
        os.path.basename(file_path).replace("tag_neuron_map_", "").replace(".pt", "")
    )

    print(f"\n" + "=" * 60)
    print(f"PROCESSING METHOD: {method_key.upper()}")
    print(f"=" * 60)

    try:
        method_data = torch.load(file_path, weights_only=False)
        unique_labels = method_data["unique_tags"]

        print(
            f"Labels: {len(unique_labels)} ({method_data.get('tag_count', '?')} tags + {method_data.get('genre_count', '?')} genres)"
        )

        # Show some examples
        print(f"Example labels:")
        for i, label in enumerate(unique_labels[:5]):
            print(f"  {i+1}. {label}")

        # Process labels for embedding
        processed_labels = preprocess_labels_for_embedding(unique_labels)

        # Show preprocessing examples
        preprocessing_count = sum(
            1
            for i in range(len(unique_labels))
            if unique_labels[i] != processed_labels[i]
        )
        print(f"Preprocessing applied to {preprocessing_count} labels")

        if preprocessing_count > 0:
            print("Preprocessing examples:")
            shown = 0
            for i in range(len(unique_labels)):
                if unique_labels[i] != processed_labels[i] and shown < 3:
                    print(f"  '{unique_labels[i]}' → '{processed_labels[i]}'")
                    shown += 1

        # Create embeddings
        print(f"Embedding {len(processed_labels)} labels...")
        label_embeddings = embed_model.encode(
            processed_labels, convert_to_tensor=True, show_progress_bar=True
        )

        # Store embeddings
        all_embeddings[method_key] = {
            "unique_tags": unique_labels,
            "processed_tags": processed_labels,
            "embeddings": label_embeddings,
            "tag_tensor": method_data["tag_tensor"],
            "metrics": method_data["metrics"],
            "is_best_method": method_data.get("is_best_method", False),
        }

        print(f"✅ {method_key}: {label_embeddings.shape}")
        print(
            f"   Mean: {label_embeddings.mean():.4f}, Std: {label_embeddings.std():.4f}"
        )

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

# Test semantic similarity for one method (best or first available)
test_method = (
    best_method
    if best_method and best_method in all_embeddings
    else list(all_embeddings.keys())[0]
)
test_data = all_embeddings[test_method]

print(f"\n" + "=" * 60)
print(f"SEMANTIC SIMILARITY TESTING ({test_method.upper()})")
print("=" * 60)

test_queries = ["action", "comedy", "drama", "thriller", "romantic"]

for query in test_queries[:3]:  # Test fewer for brevity
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_emb, test_data["embeddings"])[0]
    top_results = torch.topk(similarities, k=3)

    print(f"\nTop matches for '{query}':")
    for i, (idx, score) in enumerate(zip(top_results.indices, top_results.values)):
        label_name = test_data["unique_tags"][idx.item()]
        label_type = "🎬" if label_name.startswith("genre:") else "🏷️"
        print(f"  {i+1}. {label_type} '{label_name}' (similarity: {score.item():.4f})")

# Save embeddings for each method to separate files
print(f"\n" + "=" * 60)
print(f"SAVING EMBEDDINGS")
print(f"=" * 60)

for method_key, embedding_data in all_embeddings.items():
    # Save individual method embedding
    individual_data = {
        "method_key": method_key,
        "model_name": "all-distilroberta-v1",
        "embedding_dimension": embedding_data["embeddings"].shape[1],
        "has_genres": True,
        "preprocessing_applied": True,
        **embedding_data,
    }

    filename = f"models/tag_embeddings_{method_key}.pt"
    torch.save(individual_data, filename)
    print(f"✅ Saved {method_key} embeddings → {filename}")

# Also save a combined summary
combined_summary = {
    "methods": all_embeddings,
    "best_method": best_method,
    "model_name": "all-distilroberta-v1",
    "embedding_dimension": list(all_embeddings.values())[0]["embeddings"].shape[1],
    "has_genres": True,
}

torch.save(combined_summary, "models/tag_embeddings_summary.pt")
print(f"✅ Saved combined summary → models/tag_embeddings_summary.pt")

# Final summary
print(f"\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Model: sentence-transformers/all-distilroberta-v1")
print(f"✅ Methods processed: {len(all_embeddings)}")
for method_key, data in all_embeddings.items():
    status = "⭐ BEST" if data["is_best_method"] else ""
    print(f"   - {method_key}: {len(data['unique_tags'])} labels {status}")
print(
    f"✅ Embedding dimension: {list(all_embeddings.values())[0]['embeddings'].shape[1]}"
)
print(f"✅ Files created:")
for method_key in all_embeddings.keys():
    print(f"   - models/tag_embeddings_{method_key}.pt")
print(f"   - models/tag_embeddings_summary.pt")
