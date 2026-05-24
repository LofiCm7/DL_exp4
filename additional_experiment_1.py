import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from data_loader import SIMCLR_AUGMENTATIONS, build_simclr_pretrain_dataloader
from linear_probe import run_linear_probe_experiment
from plot import plot_pretraining_history
from test_linear_probe import run_linear_probe_test
from simclr import SimCLRModel
from utils import (
    build_run_name,
    ensure_dir,
    format_ratio_tag,
    get_device,
    resolve_output_dir,
    resolve_path,
    save_history_csv,
    seed_everything,
)


LOSS_CHOICES = ("nt_xent", "triplet", "contrastive")


class BatchTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z_i, z_j):
        if z_i.size(0) < 2:
            raise ValueError("Triplet loss requires batch_size >= 2.")

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        distances = torch.cdist(z_i, z_j, p=2)
        mask = torch.eye(z_i.size(0), device=z_i.device, dtype=torch.bool)
        positives = distances.diag()
        negatives_i = distances.masked_fill(mask, float("inf")).min(dim=1).values
        negatives_j = distances.T.masked_fill(mask, float("inf")).min(dim=1).values
        loss_i = F.relu(positives - negatives_i + self.margin)
        loss_j = F.relu(positives - negatives_j + self.margin)
        return 0.5 * (loss_i.mean() + loss_j.mean())


class BatchContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z_i, z_j):
        if z_i.size(0) < 2:
            raise ValueError("Contrastive loss requires batch_size >= 2.")

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        distances = torch.cdist(z_i, z_j, p=2)
        positive_loss = distances.diag().pow(2).mean()
        mask = ~torch.eye(z_i.size(0), device=z_i.device, dtype=torch.bool)
        negative_loss = F.relu(self.margin - distances[mask]).pow(2).mean()
        return positive_loss + negative_loss


class NTXentLossWrapper(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        if z_i.ndim != 2 or z_i.shape != z_j.shape:
            raise ValueError("z_i and z_j must have shape (batch_size, feature_dim).")
        if z_i.size(0) < 2:
            raise ValueError("NT-Xent loss requires batch_size >= 2.")

        batch_size = z_i.size(0)
        logits = torch.cat([z_i, z_j], dim=0)
        logits = F.normalize(logits, dim=1) @ F.normalize(logits, dim=1).T
        logits = logits / self.temperature
        logits.fill_diagonal_(float("-inf"))

        targets = torch.arange(batch_size, device=logits.device)
        targets = torch.cat([targets + batch_size, targets])
        return F.cross_entropy(logits, targets)


def build_pretrain_loss(name, temperature, margin):
    if name == "nt_xent":
        return NTXentLossWrapper(temperature)
    if name == "triplet":
        return BatchTripletLoss(margin)
    if name == "contrastive":
        return BatchContrastiveLoss(margin)
    raise ValueError(f"Unsupported loss '{name}'. Choices: {LOSS_CHOICES}")


def train_pretrain_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for (view_i, view_j), _ in data_loader:
        view_i = view_i.to(device)
        view_j = view_j.to(device)

        optimizer.zero_grad()
        _, z_i = model(view_i)
        _, z_j = model(view_j)
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * view_i.size(0)
        total_samples += view_i.size(0)

    return total_loss / total_samples


def run_pretraining(args, loss_name, variant_dir, device):
    pretrain_dir = ensure_dir(variant_dir / "pretrain")
    weights_dir = ensure_dir(pretrain_dir / "weights")
    curves_dir = ensure_dir(pretrain_dir / "curves")

    data_loader = build_simclr_pretrain_dataloader(
        data_root=str(args.data_root),
        image_size=args.image_size,
        batch_size=args.pretrain_batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    model = SimCLRModel(
        encoder_name=args.model,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
    ).to(device)
    criterion = build_pretrain_loss(loss_name, args.temperature, args.margin)
    optimizer = Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_weight_decay)

    history = {"epoch": [], "train_loss": []}
    best_loss = float("inf")
    checkpoint_path = weights_dir / "best_pretrain.pt"

    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss = train_pretrain_epoch(model, data_loader, criterion, optimizer, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        print(f"[pretrain][{loss_name}] Epoch [{epoch}/{args.pretrain_epochs}] train_loss={train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "model": args.model,
                    "augment": args.augment,
                    "loss_name": loss_name,
                    "encoder_state_dict": model.encoder.state_dict(),
                },
                checkpoint_path,
            )

    save_history_csv(history, curves_dir / "pretraining_history.csv")
    plot_pretraining_history(history, curves_dir / "pretraining_loss_curve.png")
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Additional Experiment 1: compare different contrastive losses."
    )
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/additional_experiment_1")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--augment", type=str, choices=SIMCLR_AUGMENTATIONS, default="simclr_v1")
    parser.add_argument("--losses", nargs="+", choices=LOSS_CHOICES, default=list(LOSS_CHOICES))
    parser.add_argument("--labeled-ratios", nargs="+", type=float, default=[0.1])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    args.data_root = resolve_path(args.data_root, must_exist=True)

    base_run_name = args.run_name or build_run_name(
        "additional_experiment_1",
        args.model,
        args.augment,
    )
    base_output_dir = resolve_output_dir(args.output_dir, base_run_name)

    print(f"Data root: {args.data_root}")
    print(f"Base run name: {base_run_name}")
    print(f"Base output dir: {base_output_dir}")

    for loss_name in args.losses:
        variant_dir = ensure_dir(base_output_dir / f"loss_{loss_name}")
        print(f"Loss variant: {loss_name}")
        print(f"Variant dir: {variant_dir}")

        pretrained_checkpoint = run_pretraining(args, loss_name, variant_dir, device)
        for labeled_ratio in args.labeled_ratios:
            label_tag = format_ratio_tag("label", labeled_ratio)
            probe_result = run_linear_probe_experiment(
                data_root=args.data_root,
                pretrained_checkpoint=pretrained_checkpoint,
                output_dir=variant_dir,
                run_name=f"linear_probe_{label_tag}",
                model=args.model,
                augment=args.augment,
                labeled_ratio=labeled_ratio,
                val_ratio=args.val_ratio,
                image_size=args.image_size,
                epochs=args.probe_epochs,
                batch_size=args.probe_batch_size,
                num_workers=args.num_workers,
                lr=args.probe_lr,
                weight_decay=args.probe_weight_decay,
                seed=args.seed,
                device=device,
                checkpoint_extra={"loss_name": loss_name},
            )
            run_linear_probe_test(
                data_root=args.data_root,
                checkpoint=probe_result["checkpoint_path"],
                output_dir=variant_dir,
                run_name=f"final_test_{label_tag}",
                model=args.model,
                batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
                device=device,
            )


if __name__ == "__main__":
    main()
