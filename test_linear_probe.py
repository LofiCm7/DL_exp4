import argparse

import torch.nn as nn

from data_loader import build_test_dataloader
from plot import plot_confusion_matrix
from simclr import load_linear_probe_model
from test import evaluate
from utils import ensure_dir, get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen linear probe on the CIFAKE test set."
    )
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="DL_exp4/output/final_test_linear_probe",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    output_dir = ensure_dir(args.output_dir)
    results_dir = ensure_dir(output_dir / "results")
    matrices_dir = ensure_dir(output_dir / "matrices")

    test_loader, class_names = build_test_dataloader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_linear_probe_model(
        args.checkpoint,
        len(class_names),
        encoder_name=args.model,
        map_location=device,
    ).to(device)

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


if __name__ == "__main__":
    main()
