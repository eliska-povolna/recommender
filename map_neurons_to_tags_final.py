import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy

from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k, hidden_dim

import re


def has_negation_words(tag):
    """Check for negation words with proper word boundaries"""
    negation_patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bwithout\b",
        r"\bisn\'t\b",
        r"\bdoesn\'t\b",
        r"\bwon\'t\b",
        r"\bcan\'t\b",
        r"\bdon\'t\b",
        r"\bdidn\'t\b",
    ]
    return any(re.search(pattern, tag) for pattern in negation_patterns)


# Load metadata
tags_df = pd.read_csv("data/tags.csv")
movies_df = pd.read_csv("data/movies.csv")
with open("data/item2index.pkl", "rb") as f:
    item2index = pickle.load(f)

# Try to load movie titles for better diagnostics
try:
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    print("Loaded movie titles for diagnostics")
except:
    movieid_to_title = {}
    print("No movie titles available - using IDs only")

index2item = {v: k for k, v in item2index.items()}
num_items = len(item2index)

# Load models
elsa = ELSA(num_items, latent_dim)
elsa.load_state_dict(torch.load("models/elsa_model_best.pt"))
elsa.eval()

sae = TopKSAE(latent_dim, hidden_dim, k)
sae.load_state_dict(torch.load("models/sae_model_r4_k32.pt"))
sae.eval()

print("Computing sparse embeddings...")
with torch.no_grad():
    embeddings = elsa.A.clone()
    batch_size = 1024
    h_list = []

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_emb = embeddings[start:end]

        batch_emb_norm = torch.nn.functional.normalize(batch_emb, dim=1)
        _, h_sparse_batch, _ = sae(batch_emb_norm)
        h_list.append(h_sparse_batch)

    h_sparse = torch.cat(h_list, dim=0)

print(f"Sparse representations shape: {h_sparse.shape}")

# EXTRACT GENRES
print("Processing movie genres...")
genre_items = defaultdict(list)

for _, row in movies_df.iterrows():
    movie_id = row["movieId"]
    if movie_id in item2index and pd.notna(row.get("genres")):
        genres = [g.strip().lower() for g in row["genres"].split("|")]
        for genre in genres:
            if genre and genre != "(no genres listed)":
                genre_items[genre].append(item2index[movie_id])

valid_genres = genre_items

print(f"Valid genres: {list(valid_genres.keys())}")

# EXTRACT TAGS
tags_df["tag"] = tags_df["tag"].str.lower()
tag_counts = tags_df["tag"].value_counts()
min_tag_occurrences = 50

# Filter tags by occurrence and quality
valid_tags = []
for tag, count in tag_counts.items():
    if (
        min_tag_occurrences <= count
        and 2 <= len(tag) <= 25
        and not tag.isdigit()
        and (tag.isalpha() or " " in tag or "-" in tag)
        and not has_negation_words(tag)
    ):
        valid_tags.append(tag)

print(f"Valid tags: {len(valid_tags)} (from {len(tag_counts)} total)")

# Create tag items dictionary directly
tag_items_dict = defaultdict(list)
for _, row in tags_df.iterrows():
    movie_id = row["movieId"]
    tag = row["tag"]
    if movie_id in item2index and tag in valid_tags:
        item_idx = item2index[movie_id]
        tag_items_dict[tag].append(item_idx)

print(f"Valid tag mappings: {len(tag_items_dict)}")

all_tag_items = {}
for tag, items in tag_items_dict.items():
    all_tag_items[f"tag:{tag}"] = items

for genre, items in valid_genres.items():
    all_tag_items[f"genre:{genre}"] = items

print(
    f"Total labels: {len(all_tag_items)} ({len(tag_items_dict)} tags + {len(valid_genres)} genres)"
)

# ===== COMPARISON OF MULTIPLE METHODS =====


def method_1_tfidf_paper(all_tag_items, sparse_activations):
    """Method 1: TF-IDF approach following the paper methodology"""
    print("Method 1: TF-IDF")

    all_labels = list(all_tag_items.keys())
    num_labels = len(all_labels)
    num_items = sparse_activations.shape[0]
    num_neurons = sparse_activations.shape[1]

    # Step 1: Build empirical joint distribution matrix [tags × items]
    print(f"Building joint distribution matrix: {num_labels} tags × {num_items} items")
    joint_distribution = torch.zeros(num_labels, num_items)

    for label_idx, (label, items) in enumerate(all_tag_items.items()):
        for item in items:
            joint_distribution[label_idx, item] = 1.0  # Binary occurrence

    # Step 2: Matrix multiplication to get tag-neuron activations
    # joint_distribution: [tags × items] @ sparse_activations: [items × neurons]
    # = tag_neuron_matrix: [tags × neurons]
    print("Computing tag-neuron activation matrix...")
    tag_neuron_matrix = torch.mm(joint_distribution, sparse_activations)

    # Normalize by number of items per tag to get average activation
    tag_counts = joint_distribution.sum(dim=1, keepdim=True)  # Number of items per tag
    tag_neuron_matrix = tag_neuron_matrix / (tag_counts + 1e-8)
    print(f"DEBUG Joint Distribution:")
    print(f"  Matrix shape: {joint_distribution.shape}")
    print(f"  Nonzero entries: {joint_distribution.nonzero().shape[0]}")

    sample_tag = list(all_tag_items.keys())[0]
    sample_items = all_tag_items[sample_tag]
    print(f"  Sample tag '{sample_tag}' has {len(sample_items)} items")
    print(f"  First 5 item indices: {sample_items[:5]}")
    print(f"  Max item index: {max(sample_items) if sample_items else 'N/A'}")
    print(f"  Expected num_items: {num_items}")

    # Step 3: Create documents for TF-IDF
    label_documents = []
    for label_idx in range(num_labels):
        neuron_activations = tag_neuron_matrix[label_idx]
        doc_words = []
        for neuron_idx, activation in enumerate(neuron_activations):
            if activation > 1e-6:  # Only include active neurons
                # Scale activation to reasonable word count (1-50 repetitions)
                word_count = max(1, min(50, int(np.sqrt(activation.item()) * 100)))
                doc_words.extend([f"n{neuron_idx}"] * word_count)
        label_documents.append(" ".join(doc_words) if doc_words else "empty")

    # Step 4: Apply TF-IDF
    tfidf = TfidfVectorizer(
        token_pattern=r"n\d+",
        max_features=num_neurons,
        lowercase=False,
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(label_documents)
    feature_names = tfidf.get_feature_names_out()

    print(
        f"TF-IDF created {len(feature_names)} features from {len(label_documents)} documents"
    )

    # Step 5: Convert back to label-neuron mapping
    label_neuron_tfidf = np.zeros((num_labels, num_neurons))
    for label_idx in range(num_labels):
        for feature_idx, tfidf_score in enumerate(
            tfidf_matrix[label_idx].toarray().flatten()
        ):
            if tfidf_score > 0:
                feature_name = feature_names[feature_idx]
                neuron_idx = int(feature_name[1:])
                if neuron_idx < num_neurons:  # Safety check
                    label_neuron_tfidf[label_idx, neuron_idx] = tfidf_score

    # Step 6: L2 normalize rows
    row_norms = np.linalg.norm(label_neuron_tfidf, axis=1, keepdims=True)
    label_neuron_tfidf = label_neuron_tfidf / (row_norms + 1e-8)

    return torch.from_numpy(label_neuron_tfidf).float(), all_labels


def method_2_centroid(all_tag_items, sparse_activations):
    """Method 2: Simple centroid prototype"""
    print("Method 2: Centroid")

    tag_vectors = []
    valid_names = []

    for label, items in all_tag_items.items():
        tag_activations = sparse_activations[items]
        centroid = tag_activations.mean(dim=0)

        if centroid.norm() > 0:
            centroid = torch.nn.functional.normalize(centroid, dim=0)
            tag_vectors.append(centroid)
            valid_names.append(label)

    return (
        torch.stack(tag_vectors) if tag_vectors else torch.empty(0, hidden_dim)
    ), valid_names


def compute_entropy_metrics(tag_vectors, tag_names):
    """Compute entropy-based evaluation metrics"""
    if len(tag_vectors) == 0:
        return {
            "avg_entropy": float("inf"),
            "std_entropy": 0.0,
            "neuron_usage": 0,
            "neuron_usage_ratio": 0.0,
            "sparsity": 0,
            "sparsity_ratio": 0.0,
            "avg_similarity": 0.0,
            "separation_score": 0.0,
            "num_tags": 0,
        }

    # 1. Individual tag entropy (lower = more focused)
    tag_entropies = []
    for vector in tag_vectors:
        probs = torch.abs(vector)
        probs = probs / (probs.sum() + 1e-8)
        probs = probs.numpy() + 1e-12
        tag_entropy = entropy(probs, base=2)
        tag_entropies.append(tag_entropy)

    # 2. Neuron usage
    neuron_usage = (tag_vectors.abs() > 1e-6).any(dim=0).sum().item()
    neuron_usage_ratio = neuron_usage / tag_vectors.shape[1]

    # 3. Sparsity
    sparsity = (tag_vectors.abs() > 1e-6).sum(dim=1).float().mean().item()
    sparsity_ratio = sparsity / tag_vectors.shape[1]

    # 4. Separation
    if len(tag_vectors) > 1:
        tag_vectors_norm = torch.nn.functional.normalize(tag_vectors, dim=1)
        similarity_matrix = torch.mm(tag_vectors_norm, tag_vectors_norm.t())
        mask = ~torch.eye(len(tag_vectors), dtype=bool)
        similarities = similarity_matrix[mask]
        avg_similarity = similarities.mean().item()
    else:
        avg_similarity = 0

    return {
        "avg_entropy": np.mean(tag_entropies),
        "std_entropy": np.std(tag_entropies),
        "neuron_usage": neuron_usage,
        "neuron_usage_ratio": neuron_usage_ratio,
        "sparsity": sparsity,
        "sparsity_ratio": sparsity_ratio,
        "avg_similarity": avg_similarity,
        "separation_score": 1 - avg_similarity,
        "num_tags": len(tag_vectors),
    }


def show_representative_movies(
    concept_name,
    concept_vector,
    sparse_activations,
    index2item,
    movieid_to_title,
    top_k=5,
):
    """Show representative movies for a concept"""
    if concept_vector.norm() == 0:
        return []

    concept_norm = torch.nn.functional.normalize(concept_vector.unsqueeze(0), dim=1)
    sparse_norm = torch.nn.functional.normalize(sparse_activations, dim=1)

    similarities = torch.mm(sparse_norm, concept_norm.t()).squeeze()
    top_indices = torch.topk(
        similarities, k=min(top_k, len(similarities)), largest=True
    ).indices

    print(f"  Most representative movies:")
    for rank, item_idx in enumerate(top_indices):
        item_idx = item_idx.item()
        if item_idx in index2item:
            movie_id = index2item[item_idx]
            movie_title = movieid_to_title.get(movie_id, f"Movie {movie_id}")
            similarity_score = similarities[item_idx].item()
            print(f"    {rank+1}. {movie_title} (sim: {similarity_score:.4f})")


# ===== TEST ALL METHODS =====
print("\n" + "=" * 80)
print("TESTING ALL METHODS")
print("=" * 80)

methods = {}

# Method 1: TF-IDF (only for original tags)
tfidf_vectors, tfidf_names = method_1_tfidf_paper(all_tag_items, h_sparse)
methods["TF-IDF"] = (tfidf_vectors, tfidf_names)

# Method 2: Centroid
centroid_vectors, centroid_names = method_2_centroid(all_tag_items, h_sparse)
methods["Centroid"] = (centroid_vectors, centroid_names)

# Evaluate all methods
print("\n" + "=" * 80)
print("ENTROPY-BASED EVALUATION")
print("=" * 80)

results = {}
for method_name, (vectors, names) in methods.items():
    print(f"\n--- {method_name} ---")
    metrics = compute_entropy_metrics(vectors, names)
    results[method_name] = metrics

    print(f"Tags: {metrics['num_tags']}")
    print(f"Avg Entropy: {metrics['avg_entropy']:.3f} ± {metrics['std_entropy']:.3f}")
    print(
        f"Neuron Usage: {metrics['neuron_usage']}/{hidden_dim} ({metrics['neuron_usage_ratio']:.1%})"
    )
    print(
        f"Avg Sparsity: {metrics['sparsity']:.1f} neurons/tag ({metrics['sparsity_ratio']:.1%})"
    )
    print(f"Separation: {metrics['separation_score']:.3f}")

# Find best method
print("\n" + "=" * 80)
print("RANKING BY COMPOSITE SCORE")
print("=" * 80)

scored_methods = []
for method_name, metrics in results.items():
    if metrics["num_tags"] > 0:
        entropy_score = 1 / (metrics["avg_entropy"] + 1)
        separation_score = metrics["separation_score"]
        sparsity_score = min(1, metrics["sparsity"] / 10)

        composite_score = (
            entropy_score * 0.4 + separation_score * 0.4 + sparsity_score * 0.2
        )
        scored_methods.append((method_name, composite_score, metrics))

scored_methods.sort(key=lambda x: x[1], reverse=True)

print("Rank | Method        | Score | Entropy | Separation | Sparsity | Tags")
print("-" * 70)
for i, (method_name, score, metrics) in enumerate(scored_methods):
    print(
        f"{i+1:4d} | {method_name:12s} | {score:.3f} | {metrics['avg_entropy']:7.3f} | "
        f"{metrics['separation_score']:10.3f} | {metrics['sparsity']:8.1f} | {metrics['num_tags']:4d}"
    )

if scored_methods:
    print(f"\n💾 SAVING METHODS")
    print(f"=" * 80)

    # Common metadata
    common_data = {
        "genre_count": len(valid_genres),
        "tag_count": len(tag_items_dict),
        "all_results": results,
        "best_method": None,
    }

    # Determine best method
    if scored_methods:
        best_method_name = scored_methods[0][0]
        best_method_key = best_method_name.lower().replace(" ", "_").replace("-", "_")
        common_data["best_method"] = best_method_key
        print(f"🏆 Best method: {best_method_name}")

    # Save each method to separate file
    for method_name, (vectors, names) in methods.items():
        method_key = method_name.lower().replace(" ", "_").replace("-", "_")

        method_data = {
            **common_data,  # Include common metadata
            "method_name": method_name,
            "method_key": method_key,
            "unique_tags": names,
            "tag_tensor": vectors,
            "metrics": results[method_name],
            "is_best_method": (method_key == common_data["best_method"]),
        }

        filename = f"models/tag_neuron_map_{method_key}.pt"
        torch.save(method_data, filename)
        print(f"✅ Saved {method_name}: {len(names)} labels → {filename}")

    # Also save a summary file with all results
    summary_data = {
        **common_data,
        "available_methods": [
            name.lower().replace(" ", "_").replace("-", "_")
            for name, _ in methods.items()
        ],
        "method_scores": {
            name.lower().replace(" ", "_").replace("-", "_"): score
            for name, score, _ in scored_methods
        },
    }
    torch.save(summary_data, "models/tag_neuron_summary.pt")
    print(f"✅ Saved summary → models/tag_neuron_summary.pt")

    print(f"\nFiles created:")
    for method_name, _ in methods.items():
        method_key = method_name.lower().replace(" ", "_").replace("-", "_")
        print(f"  - models/tag_neuron_map_{method_key}.pt")
    print(f"  - models/tag_neuron_summary.pt")

    print("\n" + "=" * 80)
    print("FANTASY TAG ANALYSIS")
    print("=" * 80)

    # Najdi fantasy tagy:
    fantasy_tags = [
        (name, items)
        for name, items in all_tag_items.items()
        if "fantasy" in name.lower()
    ]

    print(f"Found {len(fantasy_tags)} fantasy-related tags:")
    for tag_name, items in fantasy_tags[:5]:
        print(f"  {tag_name}: {len(items)} items")

        # Ukáž vzorové filmy:
        sample_movies = []
        for item_idx in items[:5]:
            if item_idx in index2item:
                movie_id = index2item[item_idx]
                movie_title = movieid_to_title.get(movie_id, f"Movie {movie_id}")
                sample_movies.append(movie_title)
        print(f"    Sample movies: {sample_movies}")

        # Zkontroluj neuron aktivace pro tento tag:
        if fantasy_tags:
            tag_idx = list(all_tag_items.keys()).index(tag_name)
            centroid_vector = (
                centroid_vectors[tag_idx] if tag_idx < len(centroid_vectors) else None
            )
            tfidf_vector = (
                tfidf_vectors[tag_idx] if tag_idx < len(tfidf_vectors) else None
            )

            if centroid_vector is not None:
                print(
                    f"    Centroid vector: max={centroid_vector.max():.4f}, norm={centroid_vector.norm():.4f}"
                )
            if tfidf_vector is not None:
                print(
                    f"    TF-IDF vector: max={tfidf_vector.max():.4f}, norm={tfidf_vector.norm():.4f}"
                )

    # Zkontroluj, jestli fantasy genre má správné filmy:
    fantasy_genre_items = all_tag_items.get("genre:fantasy", [])
    print(f"\nFantasy genre has {len(fantasy_genre_items)} items:")
    fantasy_movies = []
    for item_idx in fantasy_genre_items[:10]:
        if item_idx in index2item:
            movie_id = index2item[item_idx]
            movie_title = movieid_to_title.get(movie_id, f"Movie {movie_id}")
            fantasy_movies.append(movie_title)
    print(f"Fantasy genre movies: {fantasy_movies}")

    print("\n" + "=" * 80)
    print("GENRE PROTOTYPE ANALYSIS")
    print("=" * 80)

    # Analyzuj prototypy pro klíčové žánry:
    key_genres = [
        "fantasy",
        "sci-fi",
        "drama",
        "action",
        "thriller",
        "comedy",
        "horror",
    ]

    for genre_name in key_genres:
        genre_key = f"genre:{genre_name}"
        if genre_key in all_tag_items:
            print(f"\n--- {genre_name.upper()} GENRE ---")

            # Najdi index žánru v metodách:
            if genre_key in centroid_names:
                centroid_idx = centroid_names.index(genre_key)
                centroid_vector = centroid_vectors[centroid_idx]

                print(f"Centroid prototype movies:")
                show_representative_movies(
                    f"Centroid {genre_name}",
                    centroid_vector,
                    h_sparse,
                    index2item,
                    movieid_to_title,
                    top_k=8,
                )

            if genre_key in tfidf_names:
                tfidf_idx = tfidf_names.index(genre_key)
                tfidf_vector = tfidf_vectors[tfidf_idx]

                print(f"\nTF-IDF prototype movies:")
                show_representative_movies(
                    f"TF-IDF {genre_name}",
                    tfidf_vector,
                    h_sparse,
                    index2item,
                    movieid_to_title,
                    top_k=8,
                )

            # Zkontroluj top neurony pro tento žánr:
            if genre_key in centroid_names:
                centroid_idx = centroid_names.index(genre_key)
                centroid_vector = centroid_vectors[centroid_idx]
                top_neurons = torch.topk(centroid_vector, 10).indices.tolist()
                print(f"Centroid top neurons: {top_neurons}")

            if genre_key in tfidf_names:
                tfidf_idx = tfidf_names.index(genre_key)
                tfidf_vector = tfidf_vectors[tfidf_idx]
                top_neurons = torch.topk(tfidf_vector, 10).indices.tolist()
                print(f"TF-IDF top neurons: {top_neurons}")

    # SPECIÁLNÍ ANALÝZA PRO FANTASY vs POPULÁRNÍ FILMY:
    print("\n" + "=" * 80)
    print("FANTASY vs POPULAR MOVIES NEURON OVERLAP")
    print("=" * 80)

    # Najdi Fight Club a jiné populární filmy v h_sparse:
    popular_movies = {
        "Fight Club": 175,
        "Shawshank Redemption": 29,
        "Usual Suspects": 12,
        "Pulp Fiction": 26,
    }

    fantasy_genre_key = "genre:fantasy"
    if fantasy_genre_key in centroid_names:
        fantasy_idx = centroid_names.index(fantasy_genre_key)
        fantasy_vector = centroid_vectors[fantasy_idx]
        fantasy_top_neurons = set(torch.topk(fantasy_vector, 20).indices.tolist())

        print(f"Fantasy top 20 neurons: {sorted(fantasy_top_neurons)}")

        for movie_name, movie_idx in popular_movies.items():
            if movie_idx < h_sparse.shape[0]:
                movie_activations = h_sparse[movie_idx]
                movie_top_neurons = set(
                    torch.topk(movie_activations, 20).indices.tolist()
                )

                overlap = fantasy_top_neurons.intersection(movie_top_neurons)
                overlap_ratio = len(overlap) / len(fantasy_top_neurons)

                print(
                    f"{movie_name}: {len(overlap)}/20 neurons overlap ({overlap_ratio:.1%})"
                )
                print(f"  Overlapping neurons: {sorted(overlap)}")
                print(f"  {movie_name} top neurons: {sorted(movie_top_neurons)}")

    # KONKRÉTNÍ FANTASY FILMY ANALÝZA:
    print("\n" + "=" * 80)
    print("SPECIFIC FANTASY MOVIES NEURON ANALYSIS")
    print("=" * 80)

    fantasy_movie_indices = {
        "Lord of the Rings: Fellowship": 276,
        "Lord of the Rings: Return of the King": 355,
    }

    for movie_name, movie_idx in fantasy_movie_indices.items():
        if movie_idx < h_sparse.shape[0]:
            print(f"\n{movie_name} (Item {movie_idx}):")
            movie_activations = h_sparse[movie_idx]
            top_neurons = torch.topk(movie_activations, 10).indices.tolist()
            top_values = torch.topk(movie_activations, 10).values.tolist()

            print(f"  Top neurons: {top_neurons}")
            print(f"  Top values: {[f'{v:.4f}' for v in top_values]}")

            # Porovnej s fantasy genre neurony:
            if fantasy_genre_key in centroid_names:
                fantasy_idx = centroid_names.index(fantasy_genre_key)
                fantasy_vector = centroid_vectors[fantasy_idx]
                fantasy_top_neurons = set(
                    torch.topk(fantasy_vector, 20).indices.tolist()
                )
                movie_top_neurons = set(top_neurons)

                overlap = fantasy_top_neurons.intersection(movie_top_neurons)
                print(
                    f"  Overlap with fantasy genre: {len(overlap)}/10 neurons ({len(overlap)/10:.1%})"
                )
