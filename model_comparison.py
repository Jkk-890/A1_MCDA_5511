from sentence_transformers import SentenceTransformer
import pandas as pd
import json

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

# Save embeddings to JSON files
with open("embeddings_distilroberta.json", "w") as f:
    json.dump(embeddings1, f)

with open("embeddings_minilm.json", "w") as f:
    json.dump(embeddings2, f)

print("Embeddings saved successfully.")
