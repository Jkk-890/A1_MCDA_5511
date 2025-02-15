import csv
import umap
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from collections import defaultdict
import pyvis
from pyvis.network import Network
import numpy as np
import seaborn as sns
import branca.colormap as cm
import branca
import pandas as pd
import re
from textwrap import wrap
import json
import os
from scipy.stats import spearmanr
import optuna
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")


project_path = os.getcwd()

# Read attendees and their responses from a CSV file, replace attendees.csv with own link or file name
attendees_map = {}
with open(os.path.join(project_path, 'MCDA5511-classmates - 2025.csv'), newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)

# Create a dictionary to store embeddings for each person
person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}

# Reducing dimensionality of embedding data, scaling to coordinate domain/range
reducer = umap.UMAP(random_state=42)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)

# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())

# Plotting and annotating data points
plt.scatter(x,y)
for i, name in enumerate(label):
    plt.annotate(name, (x[i], y[i]), fontsize="8")

# Clean-up and Export
plt.title("UMAP Visualization with Random Parameters")
plt.axis('off')
plt.savefig(os.path.join(project_path,'a_visualization.png'), dpi=100)

# Standardize the embeddings
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))

# Get person names
person_names = list(person_embeddings.keys())

# Test different random seeds
seeds = [42, 100, 999]
fig, axes = plt.subplots(1, len(seeds), figsize=(20, 5))

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.3)  # Increase spacing between plots

for i, seed in enumerate(seeds):
    # Run UMAP with different seeds
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=seed)
    reduced_data = reducer.fit_transform(scaled_data)

    # Scatter plot
    x, y = reduced_data[:, 0], reduced_data[:, 1]
    axes[i].scatter(x, y, c='blue', alpha=0.5)
    
    # Add person names
    for j, name in enumerate(person_names):
        axes[i].text(x[j], y[j], name, fontsize=8, ha='right', va='bottom')

    axes[i].set_title(f"UMAP with seed {seed}")
    axes[i].axis('off')

    # Draw a vertical line between subplots (except for the last one)
    if i < len(seeds) - 1:
        axes[i].axvline(x=max(x) + 1, color='black', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

names = list(person_embeddings.keys())  # List of student names
embeddings = np.array(list(person_embeddings.values()))  # Convert dict to NumPy array

# Standardize embeddings to have mean=0 and variance=1
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

def compute_spearman_correlation(umap_2d):
    """
    Computes the average Spearman correlation between:
    (a) Cosine similarity in the original space
    (b) Euclidean distance in the UMAP 2D space.
    """
    # Compute cosine similarity in original high-dimensional space
    cosine_sim_matrix = cosine_similarity(scaled_embeddings)

    # Compute Euclidean distance in 2D UMAP space
    euclidean_dist_matrix = squareform(pdist(umap_2d, metric='euclidean'))

    spearman_values = []

    # Compute Spearman correlation for each student
    for i in range(len(names)):
        # Rank similarities for student i
        cosine_ranks = np.argsort(np.argsort(-cosine_sim_matrix[i]))  # Negative sign to rank from high to low
        euclidean_ranks = np.argsort(np.argsort(euclidean_dist_matrix[i]))  # Rank from low to high

        # Compute Spearman correlation
        rho, _ = spearmanr(cosine_ranks, euclidean_ranks)
        spearman_values.append(rho)

    return np.mean(spearman_values)  # Return the average correlation across all students

def objective(trial):
    """
    Objective function for Optuna that finds the best UMAP parameters
    by maximizing the average Spearman correlation.
    """
    # Define the search space for UMAP hyperparameters
    n_neighbors = trial.suggest_int('n_neighbors', 2, 18)
    spread = trial.suggest_uniform('spread', 0.1, 3.0)
    min_dist = trial.suggest_uniform('min_dist', 0.01, min(spread, 2.9))  # Ensures min_dist <= spread
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'canberra'])

    # Apply UMAP with sampled parameters
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, metric=metric, random_state=42)
    umap_2d = reducer.fit_transform(scaled_embeddings)

    # Compute Spearman correlation
    return compute_spearman_correlation(umap_2d)

# Run Optuna optimization
study = optuna.create_study(direction='maximize')  # Maximize Spearman correlation
study.optimize(objective, n_trials=1000)  

# Get best parameters
best_params = study.best_params
best_score = study.best_value

# Store all trials in a DataFrame
results_df = study.trials_dataframe()
results_df = results_df[['value', 'params_n_neighbors', 'params_min_dist', 'params_spread', 'params_metric']]
results_df = results_df.rename(columns={'value': 'Spearman Correlation'})
results_df = results_df.sort_values(by='Spearman Correlation', ascending=False)  # Sort by best correlation

# Print results
print("Best Parameters:", best_params)
print("Best Spearman Correlation:", best_score)

print(results_df.head())

# Retrieve the best parameters from the Optuna study
best_params = study.best_params  

# Apply UMAP with the best parameters
reducer = umap.UMAP(**best_params, random_state=42)
reduced_data = reducer.fit_transform(scaled_data)

# Plot the final visualization
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.5)

# Add names to points
for i, name in enumerate(names):
    plt.text(reduced_data[i, 0], reduced_data[i, 1], name, fontsize=9, ha='right')

plt.title("UMAP Visualization with Tuned Parameters")
plt.axis('off')
plt.show()
