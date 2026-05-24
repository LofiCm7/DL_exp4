import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from data_loader import build_test_dataloader
from net import build_model
from plot import plot_confusion_matrix
from utils import (
    ensure_dir,
    get_device,
    infer_run_name_from_checkpoint,
    resolve_output_dir,
    resolve_path,
    save_json,
)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(data_loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary"),
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline AIGC detector.")
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="DL_exp4/output/train_val/baseline_resnet18_label1_val20/weights/best_model.pt",
    )
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/final_test")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    data_root = resolve_path(args.data_root, must_exist=True)
    checkpoint_path = resolve_path(args.checkpoint, must_exist=True)
    run_name = args.run_name or infer_run_name_from_checkpoint(checkpoint_path)
    output_dir = resolve_output_dir(args.output_dir, run_name)
    results_dir = ensure_dir(output_dir / "results")
    matrices_dir = ensure_dir(output_dir / "matrices")

    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")

    test_loader, class_names = build_test_dataloader(
        data_root=str(data_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(
        f"Test Loss: {metrics['loss']:.4f} | "
        f"Test Acc: {metrics['accuracy']:.4f} | "
        f"Test F1: {metrics['f1']:.4f}"
    )

    save_json(
        {
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["f1"],
        },
        results_dir / "test_metrics.json",
    )
    plot_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        class_names,
        matrices_dir / "test_confusion_matrix.png",
    )

    print(f"Metrics json: {results_dir / 'test_metrics.json'}")
    print(f"Test matrix: {matrices_dir / 'test_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
