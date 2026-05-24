import argparse

import torch
from torch.optim import Adam

from data_loader import SIMCLR_AUGMENTATIONS, build_simclr_pretrain_dataloader
from plot import plot_pretraining_history
from simclr import NTXentLoss, SimCLRModel
from utils import (
    build_run_name,
    ensure_dir,
    get_device,
    resolve_output_dir,
    resolve_path,
    save_history_csv,
    seed_everything,
)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a SimCLR encoder on CIFAKE.")
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/simclr_pretrain")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="mobilenet_v2")
    parser.add_argument("--augment", type=str, choices=SIMCLR_AUGMENTATIONS, default="simclr_v1")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    data_root = resolve_path(args.data_root, must_exist=True)
    run_name = args.run_name or build_run_name("simclr_pretrain", args.model, args.augment)
    output_dir = resolve_output_dir(args.output_dir, run_name)
    weights_dir = ensure_dir(output_dir / "weights")
    curves_dir = ensure_dir(output_dir / "curves")

    print(f"Data root: {data_root}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")

    train_loader = build_simclr_pretrain_dataloader(
        data_root=str(data_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    model = SimCLRModel(
        encoder_name=args.model,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
    ).to(device)
    criterion = NTXentLoss(args.temperature)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"epoch": [], "train_loss": []}
    best_loss = float("inf")
    best_checkpoint_path = weights_dir / "best_pretrain.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        print(f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "model": args.model,
                    "augment": args.augment,
                    "run_name": run_name,
                    "encoder_state_dict": model.encoder.state_dict(),
                },
                best_checkpoint_path,
            )

    save_history_csv(history, curves_dir / "pretraining_history.csv")
    plot_pretraining_history(history, curves_dir / "pretraining_loss_curve.png")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Curve csv: {curves_dir / 'pretraining_history.csv'}")
    print(f"Curve figure: {curves_dir / 'pretraining_loss_curve.png'}")


if __name__ == "__main__":
    main()
