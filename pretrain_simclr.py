import argparse

import torch
from torch.optim import Adam

from data_loader import build_simclr_pretrain_dataloader
from plot import plot_pretraining_history
from simclr import NTXentLoss, SimCLRModel
from utils import ensure_dir, get_device, seed_everything


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
    parser.add_argument("--model", type=str, default="resnet18")
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

    output_dir = ensure_dir(args.output_dir)
    weights_dir = ensure_dir(output_dir / "weights")
    curves_dir = ensure_dir(output_dir / "curves")

    train_loader = build_simclr_pretrain_dataloader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
                    "encoder_state_dict": model.encoder.state_dict(),
                },
                best_checkpoint_path,
            )

    plot_pretraining_history(history, curves_dir / "pretraining_loss_curve.png")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
