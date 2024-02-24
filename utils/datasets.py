import torch
import numpy as np
import random
import os

def create_dataset_loader(path, sequence_length, batch_size, tokenizer, workers):

    # Load dataset
    dataset = PhonemesDataset(path, PhonemesDataset)

    # Collator
    def collate_fn(batch):
        x, y_d, y_p = zip(*batch)

        # Calculate lengths
        x_lengths = torch.tensor([len(x) for x in x])
        y_d_lengths = torch.tensor([len(y) for y in y_d])
        y_p_lengths = torch.tensor([len(y) for y in y_p])

        # Pad sequences
        # x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id)
        # y_d = torch.nn.utils.rnn.pad_sequence(y_d, batch_first=True, padding_value=0)
        # y_p = torch.nn.utils.rnn.pad_sequence(y_p, batch_first=True, padding_value=tokenizer.pad_token_id)
        return x, x_lengths, y_d, y_d_lengths, y_p, y_p_lengths

    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = workers, shuffle=False, pin_memory=True, drop_last=True)
    return loader

#
# Dataset Classes
# 

class GPTDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, data_len, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def generate(self):
        while True:
            # Create offsets
            offset = random.randint(0, len(self.data) - self.sequence_length)

            # Create batches
            x = torch.from_numpy(self.data[offset : offset + self.sequence_length].astype(np.int64)).long()
            y = torch.from_numpy(self.data[offset + 1 : offset + 1 + self.sequence_length].astype(np.int64)).long()

            yield (x, y)

    def __iter__(self):
        return iter(self.generate())

class PhonemesDataset(torch.utils.data.IterableDataset):
    def __init(self, path, tokenizer):
        self.tokenizer = tokenizer
        with open(path, 'r') as file:
            for line in file.readlines():
                text, phonemes = line.split('｜')
                phonemes = phonemes.split(' ')
                parsed_phonemes = []
                for p in phonemes:
                    phoneme, d = p.split(',')
                    d = int(d)
                    parsed_phonemes.append((phoneme, d))
                self.data.append((text, parsed_phonemes))
    
    def generate(self):
        while True:
            item = random.choice(self.data)
            text, phonemes = item
            
            # Prepare input
            x = torch.tensor(self.tokenizer.encode(text)).long()

            # Prepare output
            y_d = torch.tensor([d for _, d in phonemes]).long()
            y_p = torch.tensor(self.tokenizer.encode_phoneme([p for p, _ in phonemes])).long()

            return (x, y_d, y_p)
    
    def __iter__(self):
        return iter(self.generate())
        