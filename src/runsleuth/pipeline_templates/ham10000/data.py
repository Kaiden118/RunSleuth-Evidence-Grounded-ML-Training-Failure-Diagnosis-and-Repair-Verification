"""Training and validation data: labels, an image cache, random flips and loaders."""

import torch
from torch.utils.data import DataLoader, Dataset

import config


def label(diagnosis):
    """1 for a malignant diagnosis, 0 for a benign one."""
    return int(diagnosis in config.MALIGNANT)


def flip(image, generator):
    """Flip an image left-right and up-down, each with probability one half."""
    left_right, up_down = (torch.rand(2, generator=generator) < 0.5).tolist()
    if left_right:
        image = image.flip(-1)
    if up_down:
        image = image.flip(-2)
    return image


class TrainingImages(Dataset):
    """The training images in [0, 1], converted once and cached by image, flipped anew."""

    def __init__(self, images):
        self.images = images
        self.labels = [label(diagnosis) for diagnosis in images.diagnoses]
        self.cache = {}
        self.generator = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, position):
        key = self.images.image_ids[position]
        if key not in self.cache:
            self.cache[key] = self.images.image(position).float() / 255
        return flip(self.cache[key], self.generator), self.labels[position]


def training_loader(dataset, seed, epoch):
    """One epoch of shuffled, flipped training batches, drawn from the run's seed and the
    epoch alone, so that a resumed run repeats them."""
    generator = torch.Generator().manual_seed(1000 * seed + epoch)
    dataset.generator = generator
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, generator=generator)


def validation_loader(images):
    return DataLoader(images, batch_size=config.BATCH_SIZE)


def normalize(images):
    mean = torch.tensor(config.MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(config.STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std
