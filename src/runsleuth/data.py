"""FashionMNIST data loading for controlled experiments."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from runsleuth.config import TrainingConfig

FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)


def build_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic training and validation loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )

    full_training_set = datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    training_size = len(full_training_set) - config.validation_size
    split_generator = torch.Generator().manual_seed(config.seed)
    training_set, validation_set = random_split(
        full_training_set,
        [training_size, config.validation_size],
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(config.seed)
    common_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    training_loader = DataLoader(
        training_set,
        shuffle=True,
        generator=loader_generator,
        **common_options,
    )
    validation_loader = DataLoader(
        validation_set,
        shuffle=False,
        **common_options,
    )
    return training_loader, validation_loader
