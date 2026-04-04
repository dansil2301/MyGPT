from pathlib import Path
from typing import List

import numpy as np


class KNN:
    @classmethod
    def load_example_embeddings(cls):
        path = Path(__file__).parent.parent.resolve() / "NLP" / "childern_stories_embeddings.npy"
        return np.load(path)
    
    @classmethod
    def cosine_distance(cls, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @classmethod
    def _clustering(cls, centroids: np.ndarray, embeddings: np.ndarray):
        # cosine similarity (since normalized dot product works)
        similarities = np.dot(embeddings, centroids.T)

        # pick best centroid for each embedding
        cluster_ids = np.argmax(similarities, axis=1)

        # build clusters
        clusters = {i: [] for i in range(len(centroids))}
        for idx, cid in enumerate(cluster_ids):
            clusters[cid].append(embeddings[idx])

        return clusters
    
    @classmethod
    def new_centroids(cls, clusters: dict[np.ndarray], embeddings: np.ndarray) -> List:
        centroids = []
        for cluster in clusters.values():
            if len(cluster) == 0:
                centroids.append(np.random.randn(embeddings.shape[1]))
            else:
                centroids.append(np.mean(cluster, axis=0))
        
        return np.array(centroids)

    @classmethod
    def get_clusters(cls, k: int, embeddings: np.ndarray) -> dict[np.ndarray]:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        centroids = np.array(embeddings[np.random.choice(len(embeddings), k, replace=False)])
        prev_centroids = np.zeros_like(centroids)

        while not np.allclose(prev_centroids, centroids):
            prev_centroids = centroids.copy()
            clusters = cls._clustering(centroids, embeddings)
            centroids = cls.new_centroids(clusters, embeddings)

        return clusters
