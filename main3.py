from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from azure.storage.blob import BlobServiceClient
import io

# Azure Storage Connection String
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mystoragekedar;AccountKey=nIiVQTbgKUdzxeTCiPA95Hh4GqDrJrnlABfUH1vY5nnnVBlCSe7Kap40PjalItwIFYOdgX3hqb/p+AStcq8cZg==;EndpointSuffix=core.windows.net"
INPUT_CONTAINER = "input-data"
OUTPUT_CONTAINER = "embeddings"
CSV_BLOB_NAME = "MCDA5511-classmates - 2025.csv"  # Name of the CSV file in Blob Storage

# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


# Function to read CSV from Blob Storage
def read_csv_from_blob():
    blob_client = blob_service_client.get_blob_client(container=INPUT_CONTAINER, blob=CSV_BLOB_NAME)

    # Download the file as bytes
    blob_data = blob_client.download_blob()
    df = pd.read_csv(io.StringIO(blob_data.content_as_text()))

    return df


# Load dataset from Azure Blob
df = read_csv_from_blob()

# Initialize models
models = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "all-distilroberta-v1": SentenceTransformer("all-distilroberta-v1"),
    "paraphrase-MiniLM-L6-v2": SentenceTransformer("paraphrase-MiniLM-L6-v2")
}

# Create a dictionary to hold the embeddings for each model
embeddings = {}

# Compute embeddings for each model
for model_name, model in models.items():
    embeddings[model_name] = {
        row["Name"]: model.encode(row["Description"]).tolist() for _, row in df.iterrows()
    }


# Function to upload JSON to Blob Storage
def upload_json_to_blob(json_data, filename):
    blob_client = blob_service_client.get_blob_client(container=OUTPUT_CONTAINER, blob=filename)

    # Convert dict to JSON string
    json_string = json.dumps(json_data)

    # Upload JSON string as Blob
    blob_client.upload_blob(json_string, blob_type="BlockBlob", overwrite=True)
    print(f"Uploaded {filename} to {OUTPUT_CONTAINER} container.")


# Save embeddings to Azure Blob Storage
for model_name, embedding in embeddings.items():
    filename = f"embeddings_{model_name}.json"
    upload_json_to_blob(embedding, filename)

print("Embeddings for all models saved successfully to Azure Blob Storage.")
