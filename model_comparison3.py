import numpy as np
import json
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations

# Load embeddings for all models
models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1", "paraphrase-MiniLM-L6-v2"]
embeddings = {}

for model in models:
    with open(f"embeddings_{model}.json", "r") as f:
        embeddings[model] = json.load(f)

# Function to compute Spearman's rank correlation
def spearman_corr(embedding1, embedding2):
    return spearmanr(embedding1, embedding2).correlation

# Function to compute Cosine similarity
def cosine_sim(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Iterate over all combinations of model pairs
for model1, model2 in permutations(models, 2):
    print(f"Comparing {model1} and {model2}")

    # Get embeddings for the models
    embedding_list_1 = list(embeddings[model1].values())
    embedding_list_2 = list(embeddings[model2].values())

    # Ensure both embeddings are of the same size
    min_len = min(len(embedding_list_1[0]), len(embedding_list_2[0]))

    # Truncate embeddings to the smallest size
    embedding_list_1 = [embedding[:min_len] for embedding in embedding_list_1]
    embedding_list_2 = [embedding[:min_len] for embedding in embedding_list_2]

    # Compute Spearman's Rank Correlation for each dimension and average
    spearman_corrs = [spearman_corr(np.array(embedding_list_1)[:, i], np.array(embedding_list_2)[:, i]) for i in range(min_len)]
    avg_spearman = np.mean(spearman_corrs)
    print(f"Spearman's Rank Correlation: {avg_spearman}")

    # Compute Cosine Similarity for each pair of embeddings
    cosine_sims = [cosine_sim(embedding_list_1[i], embedding_list_2[i]) for i in range(len(embedding_list_1))]
    avg_cosine_sim = np.mean(cosine_sims)
    print(f"Average Cosine Similarity: {avg_cosine_sim}")
    print("-" * 50)
