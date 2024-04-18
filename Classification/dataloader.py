import os

import numpy as np
import torch
from torch.utils.data import DataLoader


def normalize_time_series(data):
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data


def zero_pad_sequence(input_tensor, pad_length):
    return torch.nn.functional.pad(input_tensor, (0, pad_length))


def calculate_padding(seq_len, patch_size):
    padding = patch_size - (seq_len % patch_size) if seq_len % patch_size != 0 else 0
    return padding


class Load_Dataset(torch.utils.data.Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_file):
        super(Load_Dataset, self).__init__()
        self.data_file = data_file

        # Load samples and labels
        x_data = data_file["samples"]  # dim: [#samples, #channels, Seq_len]

        # x_data = normalize_time_series(x_data)

        y_data = data_file.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data).squeeze()

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)

        self.x_data = x_data.float()
        self.y_data = y_data.long().squeeze() if y_data is not None else None

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


def get_datasets(DATASET_PATH, args):
    train_file = torch.load(os.path.join(DATASET_PATH, f"train.pt"))
    seq_len = train_file["samples"].shape[-1]
    required_padding = calculate_padding(seq_len, args.patch_size)

    val_file = torch.load(os.path.join(DATASET_PATH, f"val.pt"))
    test_file = torch.load(os.path.join(DATASET_PATH, f"test.pt"))

    train_dataset = Load_Dataset(train_file)
    val_dataset = Load_Dataset(val_file)
    test_dataset = Load_Dataset(test_file)

    if required_padding != 0:
        train_file["samples"] = zero_pad_sequence(train_file["samples"], required_padding)
        val_file["samples"] = zero_pad_sequence(val_file["samples"], required_padding)
        test_file["samples"] = zero_pad_sequence(test_file["samples"], required_padding)

    # in case the dataset is too small ...
    num_samples = train_dataset.x_data.shape[0]
    if num_samples < args.batch_size:
        batch_size = num_samples // 4
    else:
        batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader
