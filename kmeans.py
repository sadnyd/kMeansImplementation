import numpy as np
import random

def load_data(file_path):
    """Load locations data from the file."""
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            longitude, latitude = map(float, line.strip().split(','))
            data.append((longitude, latitude))
    return np.array(data)

def initialize_centroids(data, k):
    """Randomly initialize k centroids from the data points."""
    indices = random.sample(range(len(data)), k)
    return data[indices]

def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    clusters = []
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Recalculate centroids as the mean of all points in each cluster."""
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            # Handle empty clusters by reinitializing to a random point
            new_centroids.append(data[random.randint(0, len(data)-1)])
    return np.array(new_centroids)

def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    """Perform the K-means clustering algorithm."""
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids

    return clusters

def write_output(file_path, clusters):
    """Write the cluster labels to the output file."""
    with open(file_path, 'w') as file:
        for i, cluster in enumerate(clusters):
            file.write(f"{i} {cluster}\n")

if __name__ == "__main__":
    input_file = "place.txt"
    output_file = "clusters.txt"
    
    # Load data
    data = load_data(input_file)

    # Perform K-means clustering
    k = 3
    clusters = kmeans(data, k)

    # Write output
    write_output(output_file, clusters)

    print("Clustering completed. Output saved to clusters.txt.")
