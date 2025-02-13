from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json

path = "C:/Users/kedar/Documents/DL and NLP/As1/A1_MCDA_5511/MCDA5511-classmates - 2025.csv"

# Load dataset
df = pd.read_csv(path)

# Initialize models
model1 = SentenceTransformer("all-MiniLM-L6-v2")
model2 = SentenceTransformer("all-mpnet-base-v2")

# Compute embeddings
embeddings1 = {row["Name"]: model1.encode(row["Description"]).tolist() for _, row in df.iterrows()}
embeddings2 = {row["Name"]: model2.encode(row["Description"]).tolist() for _, row in df.iterrows()}

# Save embeddings
with open("embeddings_minilm.json", "w") as f:
    json.dump(embeddings1, f)

with open("embeddings_mpnet.json", "w") as f:
    json.dump(embeddings2, f)

print("Embeddings saved successfully.")

