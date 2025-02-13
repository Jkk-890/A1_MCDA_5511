import os
import csv
import numpy as np
from numpy.linalg import norm

def read_csv_embeddings(csv_file) -> dict:
    project_path = os.getcwd()
    embedding_dict = {}
    with open(os.path.join(project_path, csv_file), newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if len(row) > 0:
                key = row[0]
                value = np.array(row[1:], dtype=float)
                embedding_dict[key] = value
                
    return embedding_dict


origin_embeddings = read_csv_embeddings('person_embeddings.csv')
modified_embeddings = read_csv_embeddings('person_embeddings_modified.csv') 
 
for key in origin_embeddings.keys():
    if not np.array_equal(origin_embeddings[key], modified_embeddings[key]):
        cosine = np.dot(origin_embeddings[key], modified_embeddings[key]) / (norm(origin_embeddings[key]) * norm(modified_embeddings[key]))
        print(f'{key} - cosine similarity: {cosine}')