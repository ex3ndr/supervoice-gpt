import torch
import numpy as np
import random

def create_dataset_loader(path, sequence_length, batch_size, workers, dtype):

    # Load data
    data = np.memmap(path, dtype=np.uint16, mode='r')
    data_len = len(data)
    
    # Create dataset
    class GPTDataset(torch.utils.data.IterableDataset):
        def __init__(self, data, data_len, sequence_length):
            self.data = data
            self.data_len = data_len
            self.sequence_length = sequence_length

        def generate(self):
            while True:
                # Create offsets
                offset = random.randint(0, self.data_len - self.sequence_length)

                # Create batches
                x = torch.from_numpy(self.data[offset : offset + self.sequence_length].astype(np.int64)).long()
                y = torch.from_numpy(self.data[offset + 1 : offset + 1 + self.sequence_length].astype(np.int64)).long()

                yield (x, y)

        def __iter__(self):
            return iter(self.generate())

    dataset = GPTDataset(data, data_len, sequence_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = workers, shuffle=False, pin_memory=True, drop_last=True)
    return loader
            