import random
import numpy as np
from itertools import groupby
import time

def map_fn(data_point, centroids):
    """Assigns a data point to its closest centroid."""
    distances = [np.linalg.norm(np.array(data_point) - np.array(centroid)) for centroid in centroids]
    closest_centroid_index = np.argmin(distances)
    return closest_centroid_index, data_point

def reduce_fn(centroid_index_data_points):
    """Calculates the new centroid for a given cluster."""
    centroid_index, data_points = centroid_index_data_points
    new_centroid = np.mean(data_points, axis=0)
    return centroid_index, new_centroid

def has_converged(old_centroids, new_centroids, tolerance=1e-5):
    """Checks if the centroids have converged."""
    for old, new in zip(old_centroids, new_centroids):
        if np.linalg.norm(np.array(old) - np.array(new)) > tolerance:
            return False
    return True

def kmeans_mapreduce(data_points, k, max_iterations=15, max_runtime_seconds=120):
    """Performs K-means clustering with iteration and runtime limits."""

    start_time = time.time()  # Record the start time

    # Initialize centroids randomly
    centroids = random.sample(data_points, k)

    for iteration in range(max_iterations):
        # Map phase
        mapped_data = [map_fn(data_point, centroids) for data_point in data_points]

        # Simulate shuffling and sorting (grouping by centroid_index)
        mapped_data.sort(key=lambda x: x[0])

        # Reduce phase
        new_centroids = []
        for centroid_index, group in groupby(mapped_data, key=lambda x: x[0]):
            data_points_in_group = [data_point for _, data_point in group]
            new_centroid = reduce_fn((centroid_index, data_points_in_group))[1]
            new_centroids.append(new_centroid)

        # Check for convergence
        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

        # Check if max runtime has been exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime_seconds:
            print(f"Max runtime of {max_runtime_seconds} seconds reached. Stopping at iteration {iteration + 1}.")
            break

    return centroids

# Load data from the file
with open("data_points.txt", "r") as f:
    data_points = [tuple(map(float, line.strip().split(","))) for line in f]

# Apply K-means for k=5 and k=8
for k in [5, 8]:
    centroids = kmeans_mapreduce(data_points, k)
    print(f"Final centroids for k={k}: {centroids}")