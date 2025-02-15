import logging
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations  # Import permutations
import azure.functions as func

# Blob storage connection setup
connection_string = "DefaultEndpointsProtocol=https;AccountName=mystoragekedar;AccountKey=nIiVQTbgKUdzxeTCiPA95Hh4GqDrJrnlABfUH1vY5nnnVBlCSe7Kap40PjalItwIFYOdgX3hqb/p+AStcq8cZg==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
input_container = "input-data"
output_container = "output-data"
logs_container = "logs"


# Function to fetch embeddings from Blob Storage
def fetch_embeddings(blob_name):
    blob_client = blob_service_client.get_blob_client(container=input_container, blob=blob_name)
    download_stream = blob_client.download_blob()
    return json.loads(download_stream.readall())


# Function to upload results to Blob Storage
def upload_result(result, blob_name):
    blob_client = blob_service_client.get_blob_client(container=output_container, blob=blob_name)
    blob_client.upload_blob(json.dumps(result), overwrite=True)


# Function to compute Spearman's rank correlation
def spearman_corr(embedding1, embedding2):
    return spearmanr(embedding1, embedding2).correlation


# Function to compute Cosine similarity
def cosine_sim(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get blob name from query params or default to 'embeddings_model1.json'
        blob_name = req.params.get('blob') or 'embeddings_model1.json'

        # Fetch embeddings from the Blob
        embeddings = fetch_embeddings(blob_name)

        # List of models
        models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1", "paraphrase-MiniLM-L6-v2"]

        # Prepare to compute the results
        results = {}

        for model1, model2 in permutations(models, 2):
            logging.info(f"Comparing {model1} and {model2}")

            embedding_list_1 = list(embeddings[model1].values())
            embedding_list_2 = list(embeddings[model2].values())

            # Ensure both embeddings are of the same size
            min_len = min(len(embedding_list_1[0]), len(embedding_list_2[0]))

            # Truncate embeddings to the smallest size
            embedding_list_1 = [embedding[:min_len] for embedding in embedding_list_1]
            embedding_list_2 = [embedding[:min_len] for embedding in embedding_list_2]

            # Compute Spearman's Rank Correlation
            spearman_corrs = [spearman_corr(np.array(embedding_list_1)[:, i], np.array(embedding_list_2)[:, i]) for i in
                              range(min_len)]
            avg_spearman = np.mean(spearman_corrs)

            # Compute Cosine Similarity
            cosine_sims = [cosine_sim(embedding_list_1[i], embedding_list_2[i]) for i in range(len(embedding_list_1))]
            avg_cosine_sim = np.mean(cosine_sims)

            # Store results
            results[f"{model1}-{model2}"] = {
                "spearman_corr": avg_spearman,
                "cosine_similarity": avg_cosine_sim
            }

        # Upload the results to Blob Storage
        upload_result(results, 'comparison_results.json')

        return func.HttpResponse(
            f"Function executed successfully. Results uploaded to Blob Storage.",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)} wait atleast something is happeing")
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=200
        )
