import torch

# MASK_VALUE = float('-inf')
MASK_VALUE = -10000.0 # -Inf would break softmax

def create_padding_mask(lengths, max_length, device):
    batch_size = lengths.size(0)
    mask = torch.zeros(batch_size, max_length, max_length, device = device, dtype = torch.bool)
    for i in range(batch_size):
        mask[i, :lengths[i], :lengths[i]] = True

    # Convert to float
    mask = torch.where(mask, 0, MASK_VALUE)

    return mask

def create_padding_rectangle_mask(lengths1, lengths2, max_length1, max_length2, device):
    batch_size = lengths1.size(0)
    mask = torch.zeros(batch_size, max_length1, max_length2, device = device, dtype = torch.bool)
    for i in range(batch_size):
        mask[i, :lengths1[i], :lengths2[i]] = True

    # Convert to float
    mask = torch.where(mask, 0, MASK_VALUE)

    return mask

def create_padding_casual_mask(lengths, max_length, device):

    # Base mask
    batch_size = lengths.size(0)
    mask = torch.zeros(batch_size, max_length, max_length, device = device, dtype = torch.bool)
    for i in range(batch_size):
        mask[i, :lengths[i], :lengths[i]] = True

    # Casual mask
    mask = mask & torch.tril(torch.full((max_length, max_length), True, device = device), diagonal = 0)

    # Convert to float
    mask = torch.where(mask, 0, MASK_VALUE)

    return mask
    