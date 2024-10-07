import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Custom Dataset class with padding
class OneHotDataset(Dataset):
    def __init__(self, sequences_labeled, max_seq_length):
        self.sequences_labeled = sequences_labeled
        self.max_seq_length = max_seq_length - 1  # Adjust for input_ids and labels length

    def __len__(self):
        return len(self.sequences_labeled)

    def __getitem__(self, idx):
        label, seq = self.sequences_labeled[idx]
        input_ids = seq[:-1]  # All except last token
        labels = seq[1:]      # All except first token

        # Padding
        padding_length = self.max_seq_length - len(input_ids)
        input_ids_padded = input_ids + [0] * padding_length
        labels_padded = labels + [-100] * padding_length  # Use -100 to ignore padding in loss

        return {
            'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
            'labels': torch.tensor(labels_padded, dtype=torch.long),
            'label': label
        }