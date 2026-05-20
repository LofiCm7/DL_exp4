import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from data_loader import build_test_dataloader
from net import build_model
from plot import plot_confusion_matrix
from utils import ensure_dir, get_device, save_json


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline AIGC detector.")
    parser.add_argument("--data-root", type=str, default="data4")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="output/final_test")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    test_loader, meta = build_test_dataloader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(name=args.model, num_classes=len(meta["class_names"]))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"Test Loss: {metrics['loss']:.4f} | "
        f"Test Acc: {metrics['accuracy']:.4f} | "
        f"Test F1: {metrics['f1']:.4f}"
    )

    output_dir = ensure_dir(args.output_dir)
    save_json(
        {
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["f1"],
            "checkpoint": args.checkpoint,
            "model": args.model,
            "num_test_samples": meta["num_test_samples"],
        },
        output_dir / "test_metrics.json",
    )
    plot_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        meta["class_names"],
        output_dir / "test_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
