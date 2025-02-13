from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr

# File path to your CSV
path = "C:\\Users\\kedar\\Documents\\DL and NLP\\As1\\A1_MCDA_5511\\MCDA5511-classmates - 2025.csv"

# Load dataset
df = pd.read_csv(path)

# Initialize models
model1 = SentenceTransformer("all-distilroberta-v1")
model2 = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Compute embeddings for both models
embeddings1 = {row["Name"]: model1.encode(row["Description"]).tolist() for _, row in df.iterrows()}
embeddings2 = {row["Name"]: model2.encode(row["Description"]).tolist() for _, row in df.iterrows()}

# Check the dimensionality of the embeddings from both models
embedding_size_model1 = len(embeddings1[list(embeddings1.keys())[0]])
embedding_size_model2 = len(embeddings2[list(embeddings2.keys())[0]])

print(f"Embedding size for all-distilroberta-v1: {embedding_size_model1}")
print(f"Embedding size for paraphrase-MiniLM-L6-v2: {embedding_size_model2}")

# Ensure embeddings have the same dimensionality
if embedding_size_model1 != embedding_size_model2:
    print(f"Warning: Embedding sizes do not match! Truncating embeddings to the smaller size.")
    min_size = min(embedding_size_model1, embedding_size_model2)
    embeddings1 = {k: v[:min_size] for k, v in embeddings1.items()}
    embeddings2 = {k: v[:min_size] for k, v in embeddings2.items()}

# Save embeddings to JSON files
with open("embeddings_distilroberta.json", "w") as f:
    json.dump(embeddings1, f)

with open("embeddings_minilm.json", "w") as f:
    json.dump(embeddings2, f)

# Compute Spearman's Rank Correlation
names = list(df["Name"])
embedding_list_1 = [embeddings1[name] for name in names]
embedding_list_2 = [embeddings2[name] for name in names]

# Compute the Spearman correlation for each dimension of the embeddings
spearman_corrs = [spearmanr(np.array(embedding_list_1)[:, i], np.array(embedding_list_2)[:, i]) for i in range(len(embedding_list_1[0]))]

# Calculate average correlation across all dimensions
avg_spearman_corr = np.mean([corr[0] for corr in spearman_corrs])

print(f"Spearman's Rank Correlation (average across all dimensions): {avg_spearman_corr:.4f}")
