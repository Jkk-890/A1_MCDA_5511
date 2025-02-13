import numpy as np
import json
from scipy.stats import spearmanr

# Load embeddings from the saved files
with open("embeddings_distilroberta.json", "r") as f:
    embeddings1 = json.load(f)

with open("embeddings_minilm.json", "r") as f:
    embeddings2 = json.load(f)

# Compute Spearman's Rank Correlation
names = list(embeddings1.keys())
embedding_list_1 = [embeddings1[name] for name in names]
embedding_list_2 = [embeddings2[name] for name in names]

# Ensure that both embeddings have the same dimensionality
embedding_size_model1 = len(embedding_list_1[0])
embedding_size_model2 = len(embedding_list_2[0])

if embedding_size_model1 != embedding_size_model2:
    print(f"Warning: Embedding sizes do not match! Truncating embeddings to the smaller size.")
    min_size = min(embedding_size_model1, embedding_size_model2)
    embedding_list_1 = [embedding[:min_size] for embedding in embedding_list_1]
    embedding_list_2 = [embedding[:min_size] for embedding in embedding_list_2]

# Compute the Spearman correlation for each dimension of the embeddings
spearman_corrs = [spearmanr(np.array(embedding_list_1)[:, i], np.array(embedding_list_2)[:, i]) for i in range(len(embedding_list_1[0]))]

# Calculate average correlation across all dimensions
avg_spearman_corr = np.mean([corr[0] for corr in spearman_corrs])

print(f"Spearman's Rank Correlation (average across all dimensions): {avg_spearman_corr:.4f}")
