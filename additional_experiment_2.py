import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from data_loader import SIMCLR_AUGMENTATIONS, build_simclr_pretrain_dataloader
from linear_probe import run_linear_probe_experiment
from net import build_encoder
from plot import plot_pretraining_history
from simclr import NTXentLoss
from test_linear_probe import run_linear_probe_test
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


HEAD_CHOICES = ("mlp", "mlp_bn", "mlp_wide")


class ProjectionHeadVariant(nn.Module):
    def __init__(self, input_dim, projection_dim, head_name, hidden_dim=None, wide_multiplier=2.0):
        super().__init__()
        if head_name == "mlp":
            hidden_dim = input_dim if hidden_dim is None else hidden_dim
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            ]
        elif head_name == "mlp_bn":
            hidden_dim = input_dim if hidden_dim is None else hidden_dim
            layers = [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            ]
        elif head_name == "mlp_wide":
            hidden_dim = int(input_dim * wide_multiplier) if hidden_dim is None else hidden_dim
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            ]
        else:
            raise ValueError(f"Unsupported head '{head_name}'. Choices: {HEAD_CHOICES}")

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SimCLRModelWithHead(nn.Module):
    def __init__(
        self,
        encoder_name="resnet18",
        projection_dim=64,
        head_name="mlp",
        projection_hidden_dim=None,
        wide_multiplier=2.0,
    ):
        super().__init__()
        self.encoder, self.feature_dim = build_encoder(encoder_name)
        self.projection_head = ProjectionHeadVariant(
            self.feature_dim,
            projection_dim,
            head_name,
            hidden_dim=projection_hidden_dim,
            wide_multiplier=wide_multiplier,
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


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


def run_pretraining(args, head_name, variant_dir, device):
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
    model = SimCLRModelWithHead(
        encoder_name=args.model,
        projection_dim=args.projection_dim,
        head_name=head_name,
        projection_hidden_dim=args.projection_hidden_dim,
        wide_multiplier=args.wide_multiplier,
    ).to(device)
    criterion = NTXentLoss(args.temperature)
    optimizer = Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_weight_decay)

    history = {"epoch": [], "train_loss": []}
    best_loss = float("inf")
    checkpoint_path = weights_dir / "best_pretrain.pt"

    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss = train_pretrain_epoch(model, data_loader, criterion, optimizer, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        print(f"[pretrain][{head_name}] Epoch [{epoch}/{args.pretrain_epochs}] train_loss={train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "model": args.model,
                    "augment": args.augment,
                    "head_name": head_name,
                    "encoder_state_dict": model.encoder.state_dict(),
                },
                checkpoint_path,
            )

    save_history_csv(history, curves_dir / "pretraining_history.csv")
    plot_pretraining_history(history, curves_dir / "pretraining_loss_curve.png")
    return checkpoint_path


def reuse_pretrained_checkpoint(source_checkpoint, variant_dir):
    source_checkpoint = resolve_path(source_checkpoint, must_exist=True)
    pretrain_dir = ensure_dir(variant_dir / "pretrain")
    weights_dir = ensure_dir(pretrain_dir / "weights")
    curves_dir = ensure_dir(pretrain_dir / "curves")

    target_checkpoint = weights_dir / "best_pretrain.pt"
    shutil.copy2(source_checkpoint, target_checkpoint)

    source_curves_dir = source_checkpoint.parent.parent / "curves"
    for name in ("pretraining_history.csv", "pretraining_loss_curve.png"):
        source_file = source_curves_dir / name
        if source_file.exists():
            shutil.copy2(source_file, curves_dir / name)

    return target_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Additional Experiment 2: compare different projection head structures."
    )
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/additional_experiment_2")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--augment", type=str, choices=SIMCLR_AUGMENTATIONS, default="simclr_v1")
    parser.add_argument("--heads", nargs="+", choices=HEAD_CHOICES, default=list(HEAD_CHOICES))
    parser.add_argument("--pretrained-checkpoints", nargs="+", default=None)
    parser.add_argument("--labeled-ratios", nargs="+", type=float, default=[0.1])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=None)
    parser.add_argument("--wide-multiplier", type=float, default=2.0)
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

    checkpoint_map = None
    if args.pretrained_checkpoints is not None:
        if len(args.pretrained_checkpoints) != len(args.heads):
            raise ValueError("The number of --pretrained-checkpoints must match the number of --heads.")
        checkpoint_map = dict(zip(args.heads, args.pretrained_checkpoints))

    base_run_name = args.run_name or build_run_name(
        "additional_experiment_2",
        args.model,
        args.augment,
    )
    base_output_dir = resolve_output_dir(args.output_dir, base_run_name)

    print(f"Data root: {args.data_root}")
    print(f"Base run name: {base_run_name}")
    print(f"Base output dir: {base_output_dir}")

    for head_name in args.heads:
        variant_dir = ensure_dir(base_output_dir / f"head_{head_name}")
        print(f"Head variant: {head_name}")
        print(f"Variant dir: {variant_dir}")

        if checkpoint_map and head_name in checkpoint_map:
            print(f"Reuse pretrained checkpoint: {checkpoint_map[head_name]}")
            pretrained_checkpoint = reuse_pretrained_checkpoint(checkpoint_map[head_name], variant_dir)
        else:
            pretrained_checkpoint = run_pretraining(args, head_name, variant_dir, device)

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
                checkpoint_extra={"head_name": head_name},
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
