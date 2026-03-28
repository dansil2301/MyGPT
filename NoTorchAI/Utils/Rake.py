from pathlib import Path
from typing import List


class Rake:    
    def __init__(self):
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self):
        path = Path(__file__).parent.resolve() / "StopWords.txt"
        with open(file=path, mode="r", encoding="utf-8") as f:
            all_words = []
            for line in f:
                all_words.append(line.replace("\n", ""))
            return all_words
        
    def _clean_words(self, text: str) -> List[str]:
        PUNCT = ".,!?;:'\"()[]{} "

        return [
            word.strip(PUNCT)
            for word in text.lower().split()
        ]
        
    def _get_candidates(self, text: str) -> List[str]:
        spaced_text = self._clean_words(text)
        splitted_text = []

        start_id = 0
        for i, word in enumerate(spaced_text):
            if word in self.stop_words or i == len(spaced_text) - 1:
                if i == len(spaced_text) - 1:
                    i += 1

                candidate = " ".join(spaced_text[start_id:i])
                if candidate:
                    splitted_text.append(candidate)

                start_id = i + 1
        
        return splitted_text
    
    def _count_degree_freq(self, candidates: List[str]) -> dict:
        word_scores = {}

        for candidate in candidates:
            words = candidate.split()
            for word in words:
                if word in word_scores:
                    word_scores[word][0] += 1                   # freq
                    word_scores[word][1] += len(words) - 1  # degree
                else:
                    word_scores[word] = [1, len(words) - 1]
        
        return word_scores
    
    def get_key_words(self, text: str):
        candidates = self._get_candidates(text)
        word_scores = self._count_degree_freq(candidates)
        candiate_scores = {}

        for candidate in candidates:
            score = 0
            for word in candidate.split():
                score += word_scores[word][1] / word_scores[word][0]
            candiate_scores[candidate] = score

        return candiate_scores
    
    def add_stop_word(self, words: List[str]) -> None:
        self.stop_words += words
