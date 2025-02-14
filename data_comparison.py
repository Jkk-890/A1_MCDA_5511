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


def compare_embeddings(origin_embeddings, modified_embeddings) -> dict:
    distances = {}
    for key in origin_embeddings.keys():
        if not np.array_equal(origin_embeddings[key], modified_embeddings[key]):
            cosine = np.dot(origin_embeddings[key], modified_embeddings[key]) / (norm(origin_embeddings[key]) * norm(modified_embeddings[key]))
            distances[key] = cosine
            
    return distances


def print_distance_stats(distances: dict, name=""):
    cosine_min = np.min(list(distances.values()))
    cosine_max = np.max(list(distances.values()))
    
    min_key = [key for key, value in distances.items() if value == cosine_min][0]
    max_key = [key for key, value in distances.items() if value == cosine_max][0]
    mean = np.mean(list(distances.values()))
    
    print(f'\n{name.capitalize()}')
    print(f'\nMean cosine similarity: {mean}')
    print(f'Min cosine similarity - {min_key}: {cosine_min}')
    print(f'Max cosine similarity - {max_key}: {cosine_max}')
    

embeddings_path = os.path.join(os.getcwd(), 'embeddings')


origin_embeddings = read_csv_embeddings(os.path.join(embeddings_path, 'person_embeddings.csv'))
synonym_embeddings = read_csv_embeddings(os.path.join(embeddings_path, 'person_embeddings_synonym.csv')) 
antonym_embeddings = read_csv_embeddings(os.path.join(embeddings_path, 'person_embeddings_antonym.csv'))
rephrase_embeddings = read_csv_embeddings(os.path.join(embeddings_path, 'person_embeddings_rephrase.csv'))

synonym_distances = compare_embeddings(origin_embeddings, synonym_embeddings)
antonym_distances = compare_embeddings(origin_embeddings, antonym_embeddings)
rephrase_distances = compare_embeddings(origin_embeddings, rephrase_embeddings)

print_distance_stats(synonym_distances, "synonym")
print_distance_stats(antonym_distances, "antonym")
print_distance_stats(rephrase_distances, "rephrase")