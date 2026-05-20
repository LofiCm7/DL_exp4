import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data_loader import build_train_val_dataloaders
from net import build_model
from plot import plot_confusion_matrix, plot_training_history
from test import evaluate
from utils import ensure_dir, get_device, save_json, seed_everything


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    iterator = data_loader
    if tqdm is not None:
        iterator = tqdm(data_loader, desc="Train", leave=False)

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if tqdm is not None:
            iterator.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline AIGC detector.")
    parser.add_argument("--data-root", type=str, default="data4")
    parser.add_argument("--output-dir", type=str, default="output/train_val")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--labeled-ratio", type=float, default=0.1)
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
    output_dir = ensure_dir(args.output_dir)

    train_loader, val_loader, meta = build_train_val_dataloaders(
        data_root=args.data_root,
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(name=args.model, num_classes=len(meta["class_names"]))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
    best_checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "class_names": meta["class_names"],
                },
                best_checkpoint_path,
            )

    save_json(
        {
            "args": vars(args),
            "meta": meta,
            "history": history,
            "best_val_accuracy": best_accuracy,
            "best_val_f1": best_metrics["f1"] if best_metrics else None,
            "best_checkpoint": str(best_checkpoint_path),
        },
        output_dir / "training_summary.json",
    )
    plot_training_history(history, output_dir)

    if best_metrics is not None:
        plot_confusion_matrix(
            best_metrics["y_true"],
            best_metrics["y_pred"],
            meta["class_names"],
            output_dir / "best_val_confusion_matrix.png",
        )

    print(f"Saved outputs to: {Path(output_dir).resolve()}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
