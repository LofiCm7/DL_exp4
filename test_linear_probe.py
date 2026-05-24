import argparse

import torch
import torch.nn as nn

from data_loader import build_test_dataloader
from plot import plot_confusion_matrix
from simclr import load_linear_probe_model
from test import evaluate
from utils import (
    ensure_dir,
    get_device,
    infer_run_name_from_checkpoint,
    resolve_output_dir,
    resolve_path,
    save_json,
)


def run_linear_probe_test(
    *,
    data_root,
    checkpoint,
    output_dir="DL_exp4/output/final_test_linear_probe",
    run_name=None,
    model=None,
    batch_size=256,
    num_workers=4,
    image_size=32,
    device="cuda",
):
    device = device if isinstance(device, torch.device) else get_device(device)

    data_root = resolve_path(data_root, must_exist=True)
    checkpoint_path = resolve_path(checkpoint, must_exist=True)
    run_name = run_name or infer_run_name_from_checkpoint(checkpoint_path)
    output_dir = resolve_output_dir(output_dir, run_name)
    results_dir = ensure_dir(output_dir / "results")
    matrices_dir = ensure_dir(output_dir / "matrices")

    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")

    test_loader, class_names = build_test_dataloader(
        data_root=str(data_root),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = load_linear_probe_model(
        checkpoint_path,
        len(class_names),
        encoder_name=model,
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

    print(f"Metrics json: {results_dir / 'test_metrics.json'}")
    print(f"Test matrix: {matrices_dir / 'test_confusion_matrix.png'}")

    return {
        "metrics": metrics,
        "output_dir": output_dir,
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen linear probe on the CIFAKE test set."
    )
    parser.add_argument("--data-root", type=str, default="DL_exp4/data4")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="DL_exp4/output/linear_probe/linear_probe_simclr_pretrain_mobilenet_v2_simclr_v1_label1_val20/weights/best_linear_probe.pt",
    )
    parser.add_argument("--output-dir", type=str, default="DL_exp4/output/final_test_linear_probe")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    run_linear_probe_test(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
