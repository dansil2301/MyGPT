import numpy as np
from tokenizers import Tokenizer


class Batch:
    def __init__(self, datasets: dict[str: tuple], tokenizer_path: str):
        """
        Not AI comment :) just wierd architecture move
        exmaple of dict:
        {<name>: (<path>, <type: pure, instuct>, <weight>)}
        """
        self.datasets = datasets
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.mmaps = {
            name: np.memmap(path, dtype=np.int32, mode='r')
            for name, (path, _, _) in self.datasets.items()
        }

        self.weights = np.array([w for _, (_, _, w) in self.datasets.items()])
        self.weights /= self.weights.sum()  # normalize in case they don't sum to 1
        self.names = list(self.datasets.keys())

    def get_batch(self, block_size: int, batch_size: int) -> tuple[np.ndarray]:
        x_list, y_list = [], []
        
        chosen = np.random.choice(self.names, size=batch_size, p=self.weights)
        for name in chosen:
            chosen_type = self.datasets[name][1]  # extract type

            arr = self.mmaps[name]
            if chosen_type == "pure":
                x, y = self._get_correct_pure(arr, block_size)
            elif chosen_type == "instruct":
                x, y = self._get_correct_instruct(arr, block_size)
            else:
                raise ValueError("Not such type found")

            x_list.append(x)
            y_list.append(y)
        return np.stack(x_list), np.stack(y_list)
    
    def _get_correct_pure(self, arr: np.ndarray, block_size: int) -> tuple[np.ndarray]:
        idx = np.random.randint(0, len(arr) - block_size - 1)
        x = arr[idx:idx + block_size]
        y = arr[idx + 1:idx + block_size + 1]
        return x, y

    def _get_correct_instruct(self, arr: np.ndarray, block_size: int) -> tuple[np.ndarray]:
        assitant_token = self.tokenizer.encode("<|assistant|>").ids[0]  # should always be a single token
        while True:
            idx = np.random.randint(0, len(arr) - block_size - 1)
            x = arr[idx:idx + block_size]
            if (x == assitant_token).any():
                continue
            y = arr[idx + 1:idx + block_size + 1]

            return x, y
    