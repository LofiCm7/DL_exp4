import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from data_loader import build_train_val_dataloaders
from plot import plot_confusion_matrix, plot_training_history
from simclr import LinearProbeModel, load_pretrained_encoder
from test import evaluate
from utils import ensure_dir, get_device, seed_everything


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
    parser = argparse.ArgumentParser(
        description="Train a frozen linear probe on top of a SimCLR encoder."
    )
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="DL_exp4/output/simclr_pretrain/best_pretrain.pt",
    )
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/linear_probe")
    parser.add_argument("--model", type=str, default=None)
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
    weights_dir = ensure_dir(output_dir / "weights")
    curves_dir = ensure_dir(output_dir / "curves")
    matrices_dir = ensure_dir(output_dir / "matrices")

    train_loader, val_loader, class_names = build_train_val_dataloaders(
        data_root=args.data_root,
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    encoder, feature_dim, model_name = load_pretrained_encoder(
        args.pretrained_checkpoint,
        encoder_name=args.model,
    )
    model = LinearProbeModel(encoder, feature_dim, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    best_checkpoint_path = weights_dir / "best_linear_probe.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
            torch.save(
                {
                    "model": model_name,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "classifier_state_dict": model.classifier.state_dict(),
                },
                best_checkpoint_path,
            )

    plot_training_history(history, curves_dir / "training_curves.png")
    if best_metrics is not None:
        plot_confusion_matrix(
            best_metrics["y_true"],
            best_metrics["y_pred"],
            class_names,
            matrices_dir / "best_val_confusion_matrix.png",
        )

    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
