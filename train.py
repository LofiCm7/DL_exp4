import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from data_loader import build_train_val_dataloaders
from net import build_model
from plot import plot_confusion_matrix, plot_training_history
from test import evaluate
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


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline AIGC detector.")
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/train_val")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--labeled-ratio", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    data_root = resolve_path(args.data_root, must_exist=True)
    run_name = args.run_name or build_run_name(
        "baseline",
        args.model,
        format_ratio_tag("label", args.labeled_ratio),
        format_ratio_tag("val", args.val_ratio),
    )
    output_dir = resolve_output_dir(args.output_dir, run_name)
    weights_dir = ensure_dir(output_dir / "weights")
    curves_dir = ensure_dir(output_dir / "curves")
    matrices_dir = ensure_dir(output_dir / "matrices")

    train_loader, val_loader, class_names = build_train_val_dataloaders(
        data_root=str(data_root),
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    best_accuracy = -1.0
    best_metrics = None
    best_checkpoint_path = weights_dir / "best_model.pt"

    print(f"Data root: {data_root}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_metrics = val_metrics
            torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    save_history_csv(history, curves_dir / "training_history.csv")
    plot_training_history(history, curves_dir / "training_curves.png")
    if best_metrics is not None:
        plot_confusion_matrix(
            best_metrics["y_true"],
            best_metrics["y_pred"],
            class_names,
            matrices_dir / "best_val_confusion_matrix.png",
        )

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Curve csv: {curves_dir / 'training_history.csv'}")
    print(f"Curve figure: {curves_dir / 'training_curves.png'}")
    print(f"Val matrix: {matrices_dir / 'best_val_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
