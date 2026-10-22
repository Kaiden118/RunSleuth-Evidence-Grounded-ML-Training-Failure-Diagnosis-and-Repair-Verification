"""Training and validation data: the hospitals trained on, loaders and normalization."""

import torch
from torch.utils.data import DataLoader, Dataset

import config


class HospitalSubset(Dataset):
    """The patches from the given hospitals, in the order of the patches they come from."""

    def __init__(self, patches, hospitals):
        self.patches = patches
        self.indices = [
            index for index, hospital in enumerate(patches.hospitals) if hospital in hospitals
        ]
        self.labels = [patches.labels[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        return self.patches.image(self.indices[position]), self.labels[position]


def training_set(patches):
    return HospitalSubset(patches, config.TRAINING_HOSPITALS)


def training_loader(dataset, seed, epoch):
    """One epoch of training batches, shuffled by the run's seed and the epoch alone.

    A resumed run thus shuffles every epoch as the uninterrupted run did.
    """
    generator = torch.Generator().manual_seed(1000 * seed + epoch)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, generator=generator)


def validation_loader(patches):
    return DataLoader(patches, batch_size=config.BATCH_SIZE)


def normalize(images):
    mean = torch.tensor(config.MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(config.STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std
