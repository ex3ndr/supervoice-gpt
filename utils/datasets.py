import torch
import numpy as np
import random
import os

def create_dataset_loader(path, sequence_length, batch_size, tokenizer, workers):

    # Load dataset
    dataset = PhonemesDataset(path, tokenizer, sequence_length)

    # Collator
    def collate_fn(batch):
        B = len(batch)
        x, y_d, y_p, y_pi = zip(*batch)

        # Calculate lengths
        x_lengths = torch.tensor([len(x) for x in x])
        y_lengths = torch.tensor([len(y) - 1 for y in y_d])

        # Create targets
        t_d = torch.zeros(B, sequence_length, dtype = torch.int64)
        t_p = torch.zeros(B, sequence_length, dtype = torch.int64)
        t_pi = torch.zeros(B, sequence_length, dtype = torch.int64)
        for i in range(B):
            t_d[i, :len(y_d[i]) - 1] = y_d[i][1:]
            t_p[i, :len(y_p[i]) - 1] = y_p[i][1:]
            t_pi[i, :len(y_pi[i]) - 1] = y_pi[i][1:]

        # Padded tensors
        x_padded = torch.IntTensor(B, sequence_length)
        y_t_padded = torch.IntTensor(B, sequence_length)
        y_d_padded = torch.IntTensor(B, sequence_length)
        y_pi_padded = torch.IntTensor(B, sequence_length)
        x_padded.zero_()
        y_t_padded.zero_()
        y_d_padded.zero_()
        y_pi_padded.zero_()
        for i in range(B):
            x_padded[i, :len(x[i])] = x[i]
            y_t_padded[i, :len(y_p[i]) - 1] = y_p[i][:-1]
            y_d_padded[i, :len(y_d[i]) - 1] = y_d[i][:-1]
            y_pi_padded[i, :len(y_pi[i]) - 1] = y_pi[i][:-1]

        return x_padded, x_lengths, y_t_padded, y_d_padded, y_pi_padded, y_lengths, t_p, t_d, t_pi

    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = workers, shuffle=False, pin_memory=True, drop_last=True, collate_fn = collate_fn)
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
    def __init__(self, path, tokenizer, max_sequence):
        self.tokenizer = tokenizer
        self.data = []
        self.max_sequence = max_sequence
        with open(path, 'r') as file:
            for line in file.readlines():
                text, phonemes = line.split('｜')
                phonemes = phonemes.split(' ')
                parsed_phonemes = []
                for p in phonemes:
                    phoneme, d, pitch = p.split(',')
                    d = int(d)
                    pitch = int(pitch)
                    parsed_phonemes.append((phoneme, d, pitch))
                self.data.append((text, parsed_phonemes))
    
    def generate(self):
        while True:
            item = random.choice(self.data)
            text, phonemes = item
            
            # Prepare input
            input_tokens = self.tokenizer.encode(text) if random.random() < 0.3 else self.tokenizer.encode_sample(text) # 30% chance of using optimal tokenization
            x = torch.tensor([self.tokenizer.sequence_begin_token_id] + input_tokens + [self.tokenizer.sequence_end_token_id]).int()

            # Prepare output
            y_p = torch.tensor([self.tokenizer.sequence_begin_token_id] + self.tokenizer.encode_phonemes([p for p, _, _ in phonemes]) + [self.tokenizer.sequence_end_token_id]).int()
            y_d = torch.tensor([0] + [(d + 1) for _, d, _ in phonemes] + [0]).int()
            y_pi = torch.tensor([0] + [(pitch + 1) for _, _, pitch in phonemes] + [0]).int()

            # Check if sequence is too long
            if len(x) > self.max_sequence or len(y_p) > self.max_sequence:
                continue

            yield (x, y_d, y_p, y_pi)
    
    def __iter__(self):
        return iter(self.generate())
        