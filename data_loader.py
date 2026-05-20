from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(image_size: int = 32):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, test_transform


def _sample_labeled_indices(targets, labeled_ratio: float, seed: int):
    if not 0 < labeled_ratio <= 1:
        raise ValueError("labeled_ratio must be in the range (0, 1].")

    generator = torch.Generator().manual_seed(seed)
    class_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        class_to_indices[label].append(index)

    sampled_indices = []
    for label in sorted(class_to_indices):
        indices = torch.tensor(class_to_indices[label], dtype=torch.long)
        perm = torch.randperm(len(indices), generator=generator)
        shuffled = indices[perm]
        sample_count = max(1, int(len(shuffled) * labeled_ratio))
        sampled_indices.extend(shuffled[:sample_count].tolist())

    sampled_indices.sort()
    return sampled_indices


def _split_train_val_indices(targets, val_ratio: float, seed: int):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in the range (0, 1).")

    generator = torch.Generator().manual_seed(seed)
    class_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        class_to_indices[label].append(index)

    train_indices = []
    val_indices = []

    for label in sorted(class_to_indices):
        indices = torch.tensor(class_to_indices[label], dtype=torch.long)
        perm = torch.randperm(len(indices), generator=generator)
        shuffled = indices[perm]
        if len(shuffled) == 1:
            val_count = 1
        else:
            val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * val_ratio)))

        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def build_train_val_datasets(
    data_root: str,
    labeled_ratio: float,
    val_ratio: float,
    image_size: int,
    seed: int,
):
    train_transform, eval_transform = build_transforms(image_size=image_size)

    full_train_dataset = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=train_transform,
    )
    full_eval_dataset = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=eval_transform,
    )

    full_train_indices, val_indices = _split_train_val_indices(
        full_train_dataset.targets,
        val_ratio=val_ratio,
        seed=seed,
    )
    sampled_train_indices = _sample_labeled_indices(
        [full_train_dataset.targets[index] for index in full_train_indices],
        labeled_ratio=labeled_ratio,
        seed=seed,
    )
    train_indices = [full_train_indices[index] for index in sampled_train_indices]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_eval_dataset, val_indices)

    class_names = full_train_dataset.classes
    meta = {
        "class_names": class_names,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_full_train_samples": len(full_train_dataset),
        "labeled_ratio": labeled_ratio,
        "val_ratio": val_ratio,
    }
    return train_dataset, val_dataset, meta


def build_test_dataset(data_root: str, image_size: int):
    _, eval_transform = build_transforms(image_size=image_size)
    test_dataset = datasets.ImageFolder(
        root=f"{data_root}/test",
        transform=eval_transform,
    )
    meta = {
        "class_names": test_dataset.classes,
        "num_test_samples": len(test_dataset),
    }
    return test_dataset, meta


def build_train_val_dataloaders(
    data_root: str,
    labeled_ratio: float = 0.1,
    val_ratio: float = 0.2,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
):
    train_dataset, val_dataset, meta = build_train_val_datasets(
        data_root=data_root,
        labeled_ratio=labeled_ratio,
        val_ratio=val_ratio,
        image_size=image_size,
        seed=seed,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, meta


def build_test_dataloader(
    data_root: str,
    image_size: int = 32,
    batch_size: int = 256,
    num_workers: int = 4,
):
    test_dataset, meta = build_test_dataset(
        data_root=data_root,
        image_size=image_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, meta
