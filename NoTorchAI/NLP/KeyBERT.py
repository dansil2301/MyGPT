from pathlib import Path
from typing import List

import numpy as np
from tokenizers import Tokenizer


class KeyBERT:
    def __init__(self, tokenizer_path: str):
        self.embeddings = self._load_embeddings()
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self):
        path = Path(__file__).parent.resolve() / "StopWords.txt"
        with open(file=path, mode="r", encoding="utf-8") as f:
            all_words = []
            for line in f:
                all_words.append(line.replace("\n", ""))
            return all_words

    def _load_embeddings(self):
        path = Path(__file__).parent.resolve() / "childern_stories_embeddings.npy"
        return np.load(path)
    
    def _vectorize_text(self, text: str) -> List:
        encoded = self.tokenizer.encode(text).ids
        vectorized_tokens = []
        
        for token in encoded:
            vectorized_tokens.append(self.embeddings[token])

        return np.array(vectorized_tokens)
    
    def _mean_pool(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return np.zeros(self.embeddings.shape[1])
        return np.mean(vectors, axis=0)
    
    def _vector_len(self, vector: np.ndarray) -> float:
        return np.sqrt(np.sum(vector ** 2))
    
    def _calculate_cos_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        v1_len = self._vector_len(vector1)
        v2_len = self._vector_len(vector2)
        if v1_len == 0 or v2_len == 0:
            return 0.0
        return float(np.dot(vector1, vector2) / (v1_len * v2_len))
    
    def _get_candidate_chunks(self, text: str) -> List[List[str]]:
        PUNCT = set(".,!?;:'\"()[]{}-")
        chunks = []
        current_chunk = []

        for word in text.split():
            clean = word.strip("".join(PUNCT)).lower()

            if not clean or clean in self.stop_words or all(c in PUNCT for c in word):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                current_chunk.append(clean)

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    def get_key_words(self, text: str, min_ngram: int, max_ngram: int) -> dict:
        vectorized_tokens = self._vectorize_text(text)
        base_vector = self._mean_pool(vectorized_tokens)

        sntences = text.split(".")
        candidates = {}

        for ngram in range(min_ngram, max_ngram + 1):
            for i in range(len(sntences) - ngram + 1):
                c_ngram = " ".join(sntences[i:i+ngram]).strip()
                c_vectorized_tokens = self._vectorize_text(c_ngram)
                vector = self._mean_pool(c_vectorized_tokens)
                candidates[c_ngram] = self._calculate_cos_similarity(base_vector, vector)

        return candidates
