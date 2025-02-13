from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# Path to your dataset
path = "C:/Users/kedar/Documents/DL and NLP/As1/A1_MCDA_5511/MCDA5511-classmates - 2025.csv"

# Load dataset
df = pd.read_csv(path)

# Initialize models
models = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "all-distilroberta-v1": SentenceTransformer("all-distilroberta-v1"),
    "paraphrase-MiniLM-L6-v2": SentenceTransformer("paraphrase-MiniLM-L6-v2")
}

# Create a dictionary to hold the embeddings for each model
embeddings = {}

# Compute embeddings for each model and save them
for model_name, model in models.items():
    embeddings[model_name] = {row["Name"]: model.encode(row["Description"]).tolist() for _, row in df.iterrows()}

# Save embeddings to JSON files
for model_name, embedding in embeddings.items():
    with open(f"embeddings_{model_name}.json", "w") as f:
        json.dump(embedding, f)

print("Embeddings for all models saved successfully.")
