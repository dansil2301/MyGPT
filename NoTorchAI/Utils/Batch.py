import numpy as np


class Batch:
    def __init__(self, datasets: dict[str: tuple]):
        self.datasets = datasets

        self.mmaps = {
            name: np.memmap(path, dtype=np.int32, mode='r')
            for name, (path, _) in self.datasets.items()
        }

        self.weights = np.array([w for _, (_, w) in self.datasets.items()])
        self.weights /= self.weights.sum()  # normalize in case they don't sum to 1
        self.names = list(self.datasets.keys())

    def get_batch(self, block_size, batch_size):
        x_list, y_list = [], []
        
        chosen = np.random.choice(self.names, size=batch_size, p=self.weights)
        for name in chosen:
            arr = self.mmaps[name]
            idx = np.random.randint(0, len(arr) - block_size - 1)
            x_list.append(arr[idx:idx + block_size])
            y_list.append(arr[idx + 1:idx + block_size + 1])
        return np.stack(x_list), np.stack(y_list)
