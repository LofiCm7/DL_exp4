from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
SIMCLR_AUGMENTATIONS = ("simclr_v1", "simclr_v2")


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def build_visualization_transform(image_size=32):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_transforms(image_size=32):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, eval_transform


def build_simclr_view_transform(image_size=32, augment="simclr_v1"):
    if augment == "simclr_v1":
        color_jitter = transforms.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(15)], p=0.3),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    if augment == "simclr_v2":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=0,
                            translate=(0.15, 0.15),
                            scale=(0.8, 1.2),
                            shear=12,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomEqualize(p=0.3),
                transforms.RandomApply([transforms.RandomPosterize(bits=3)], p=0.4),
                transforms.RandomSolarize(threshold=128, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    raise ValueError(
        f"Unsupported augment '{augment}'. Choices: {', '.join(SIMCLR_AUGMENTATIONS)}"
    )


def build_simclr_transform(image_size=32, augment="simclr_v1"):
    return TwoCropTransform(build_simclr_view_transform(image_size, augment))


def _sample_labeled_indices(targets, labeled_ratio, seed):
    if not 0 < labeled_ratio <= 1:
        raise ValueError("labeled_ratio must be in the range (0, 1].")

    class_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        class_to_indices[label].append(index)

    generator = torch.Generator().manual_seed(seed)
    sampled_indices = []
    for label in sorted(class_to_indices):
        indices = torch.tensor(class_to_indices[label], dtype=torch.long)
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        count = max(1, int(len(shuffled) * labeled_ratio))
        sampled_indices.extend(shuffled[:count].tolist())

    sampled_indices.sort()
    return sampled_indices


def _split_train_val_indices(targets, val_ratio, seed):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in the range (0, 1).")

    class_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        class_to_indices[label].append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    for label in sorted(class_to_indices):
        indices = torch.tensor(class_to_indices[label], dtype=torch.long)
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        if len(shuffled) == 1:
            val_count = 1
        else:
            val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * val_ratio)))
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def _build_train_val_datasets(data_root, labeled_ratio, val_ratio, image_size, seed):
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=train_transform,
    )
    eval_dataset = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=eval_transform,
    )

    train_indices, val_indices = _split_train_val_indices(
        train_dataset.targets,
        val_ratio,
        seed,
    )
    labeled_indices = _sample_labeled_indices(
        [train_dataset.targets[index] for index in train_indices],
        labeled_ratio,
        seed,
    )
    train_indices = [train_indices[index] for index in labeled_indices]

    return (
        Subset(train_dataset, train_indices),
        Subset(eval_dataset, val_indices),
        train_dataset.classes,
    )


def build_train_val_dataloaders(
    data_root,
    labeled_ratio=0.1,
    val_ratio=0.2,
    image_size=32,
    batch_size=128,
    num_workers=4,
    seed=42,
):
    train_dataset, val_dataset, class_names = _build_train_val_datasets(
        data_root,
        labeled_ratio,
        val_ratio,
        image_size,
        seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, class_names


def build_simclr_pretrain_dataloader(
    data_root,
    image_size=32,
    batch_size=256,
    num_workers=4,
    augment="simclr_v1",
):
    dataset = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=build_simclr_transform(image_size, augment),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )


def build_test_dataloader(
    data_root,
    image_size=32,
    batch_size=256,
    num_workers=4,
):
    _, eval_transform = build_transforms(image_size)
    dataset = datasets.ImageFolder(
        root=f"{data_root}/test",
        transform=eval_transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return data_loader, dataset.classes
