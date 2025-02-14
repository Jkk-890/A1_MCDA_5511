import json
import numpy as np
from scipy.stats import spearmanr
from sentence_transformers.util import cos_sim

# Load embeddings
with open("embeddings_minilm.json", "r") as f:
    embeddings_minilm = json.load(f)

with open("embeddings_mpnet.json", "r") as f:
    embeddings_mpnet = json.load(f)

# Convert embeddings to NumPy arrays
embeddings_minilm = {k: np.array(v) for k, v in embeddings_minilm.items()}
embeddings_mpnet = {k: np.array(v) for k, v in embeddings_mpnet.items()}

# Choose reference person
reference_person = "Kedar Gaikwad"

# Compute cosine similarity scores
similarities_minilm = {
    name: cos_sim(embeddings_minilm[reference_person], emb).item()
    for name, emb in embeddings_minilm.items() if name != reference_person
}

similarities_mpnet = {
    name: cos_sim(embeddings_mpnet[reference_person], emb).item()
    for name, emb in embeddings_mpnet.items() if name != reference_person
}

# Rank classmates by similarity
ranks_minilm = sorted(similarities_minilm, key=similarities_minilm.get, reverse=True)
ranks_mpnet = sorted(similarities_mpnet, key=similarities_mpnet.get, reverse=True)

# Convert rankings to numerical positions
rank_positions_minilm = [ranks_minilm.index(name) for name in similarities_minilm.keys()]
rank_positions_mpnet = [ranks_mpnet.index(name) for name in similarities_mpnet.keys()]

# Compute Spearman's rank correlation
spearman_corr, _ = spearmanr(rank_positions_minilm, rank_positions_mpnet)

print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
