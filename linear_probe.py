import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from data_loader import SIMCLR_AUGMENTATIONS, build_train_val_dataloaders
from plot import plot_confusion_matrix, plot_training_history
from simclr import LinearProbeModel, load_pretrained_encoder
from test import evaluate
from utils import (
    build_run_name,
    ensure_dir,
    format_ratio_tag,
    get_device,
    infer_run_name_from_checkpoint,
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


def run_linear_probe_experiment(
    *,
    data_root,
    pretrained_checkpoint,
    output_dir="DL_exp4/output/linear_probe",
    run_name=None,
    model=None,
    augment="simclr_v1",
    labeled_ratio=0.01,
    val_ratio=0.2,
    image_size=32,
    epochs=10,
    batch_size=128,
    num_workers=4,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    device="cuda",
    checkpoint_extra=None,
):
    seed_everything(seed)
    device = device if isinstance(device, torch.device) else get_device(device)

    data_root = resolve_path(data_root, must_exist=True)
    pretrained_checkpoint = resolve_path(pretrained_checkpoint, must_exist=True)
    pretrain_run_name = infer_run_name_from_checkpoint(pretrained_checkpoint)
    run_name = run_name or build_run_name(
        "linear_probe",
        pretrain_run_name,
        format_ratio_tag("label", labeled_ratio),
        format_ratio_tag("val", val_ratio),
    )
    output_dir = resolve_output_dir(output_dir, run_name)
    weights_dir = ensure_dir(output_dir / "weights")
    curves_dir = ensure_dir(output_dir / "curves")
    matrices_dir = ensure_dir(output_dir / "matrices")

    print(f"Data root: {data_root}")
    print(f"Pretrained checkpoint: {pretrained_checkpoint}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")

    train_loader, val_loader, class_names = build_train_val_dataloaders(
        data_root=str(data_root),
        labeled_ratio=labeled_ratio,
        val_ratio=val_ratio,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    encoder, feature_dim, model_name = load_pretrained_encoder(
        pretrained_checkpoint,
        encoder_name=model,
        map_location=device,
    )
    model = LinearProbeModel(encoder, feature_dim, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)

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

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_metrics = val_metrics
            checkpoint = {
                "model": model_name,
                "augment": augment,
                "run_name": run_name,
                "encoder_state_dict": model.encoder.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
            }
            if checkpoint_extra:
                checkpoint.update(checkpoint_extra)
            torch.save(checkpoint, best_checkpoint_path)

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

    return {
        "checkpoint_path": best_checkpoint_path,
        "output_dir": output_dir,
        "run_name": run_name,
        "history": history,
        "best_metrics": best_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a frozen linear probe on top of a SimCLR encoder."
    )
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="DL_exp4/output/simclr_pretrain/simclr_pretrain_mobilenet_v2_simclr_v1/weights/best_pretrain.pt",
    )
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/linear_probe")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--augment", type=str, choices=SIMCLR_AUGMENTATIONS, default="simclr_v1")
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
    run_linear_probe_experiment(
        data_root=args.data_root,
        pretrained_checkpoint=args.pretrained_checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        model=args.model,
        augment=args.augment,
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
