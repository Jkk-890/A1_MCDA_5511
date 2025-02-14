import json
from sentence_transformers import SentenceTransformer

# Initialize the model globally
model = None


def init():
    global model
    # Load the model (this assumes the model is uploaded as 'all-MiniLM-L6-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')


def run(input_data):
    try:
        # Parse input JSON
        data = json.loads(input_data)

        # Assuming input_data contains a list of descriptions
        descriptions = data["descriptions"]

        # Generate embeddings for the descriptions
        embeddings = model.encode(descriptions)

        # Return the embeddings as a JSON response
        return json.dumps({"embeddings": embeddings.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
