from pathlib import Path

import numpy as np


class PCA:
    @classmethod
    def load_example_embeddings(cls):
        path = Path(__file__).parent.parent.resolve() / "NLP" / "childern_stories_embeddings.npy"
        return np.load(path)

    @classmethod
    def reduce_dimensions(cls, embeddings: np.ndarray, dimensions: int = None) -> np.ndarray:
        # Center Data
        feature_mean = np.mean(embeddings, axis=0)
        centered = embeddings - feature_mean
        
        # cov matrix
        n = centered.shape[0]
        cov = (1/n) * centered.T @ centered

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Keep top components (e.g. 95% variance explained)
        if not dimensions:
            explained = eigenvalues / eigenvalues.sum()
            dimensions = np.searchsorted(np.cumsum(explained), 0.95) + 1
        W = eigenvectors[:, :dimensions]

        # Project
        reduced = centered @ W
        return reduced
