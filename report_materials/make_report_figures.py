import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
REPORT_DIR = Path(__file__).resolve().parent
RAW_DIR = REPORT_DIR / "images" / "raw"
GENERATED_DIR = REPORT_DIR / "images" / "generated"

CONFIGS = [
    {
        "code": "r0.01",
        "name": "ResNet18 1%",
        "baseline_history": OUTPUT_DIR / "train_val" / "baseline_resnet18_label1_val20" / "curves" / "training_history.csv",
        "simclr_history": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label1_val20" / "curves" / "training_history.csv",
        "baseline_test": OUTPUT_DIR / "final_test" / "baseline_resnet18_label1_val20" / "results" / "test_metrics.json",
        "simclr_test": OUTPUT_DIR / "final_test_linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label1_val20" / "results" / "test_metrics.json",
    },
    {
        "code": "r0.1",
        "name": "ResNet18 10%",
        "baseline_history": OUTPUT_DIR / "train_val" / "baseline_resnet18_label10_val20" / "curves" / "training_history.csv",
        "simclr_history": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label10_val20" / "curves" / "training_history.csv",
        "baseline_test": OUTPUT_DIR / "final_test" / "baseline_resnet18_label10_val20" / "results" / "test_metrics.json",
        "simclr_test": OUTPUT_DIR / "final_test_linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label10_val20" / "results" / "test_metrics.json",
    },
    {
        "code": "m0.01",
        "name": "MobileNetV2 1%",
        "baseline_history": OUTPUT_DIR / "train_val" / "baseline_mobilenet_v2_label1_val20" / "curves" / "training_history.csv",
        "simclr_history": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_mobilenet_v2_simclr_v1_label1_val20" / "curves" / "training_history.csv",
        "baseline_test": OUTPUT_DIR / "final_test" / "baseline_mobilenet_v2_label1_val20" / "results" / "test_metrics.json",
        "simclr_test": OUTPUT_DIR / "final_test_linear_probe" / "linear_probe_simclr_pretrain_mobilenet_v2_simclr_v1_label1_val20" / "results" / "test_metrics.json",
    },
    {
        "code": "m0.1",
        "name": "MobileNetV2 10%",
        "baseline_history": OUTPUT_DIR / "train_val" / "baseline_mobilenet_v2_label10_val20" / "curves" / "training_history.csv",
        "simclr_history": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_mobilenet_v2_simclr_v1_label10_val20" / "curves" / "training_history.csv",
        "baseline_test": OUTPUT_DIR / "final_test" / "baseline_mobilenet_v2_label10_val20" / "results" / "test_metrics.json",
        "simclr_test": OUTPUT_DIR / "final_test_linear_probe" / "linear_probe_simclr_pretrain_mobilenet_v2_simclr_v1_label10_val20" / "results" / "test_metrics.json",
    },
]

EXP1_BASE_DIR = OUTPUT_DIR / "additional_experiment_1" / "additional_experiment_1_resnet18_simclr_v1"
EXP1_LOSS_CONFIGS = [
    {
        "name": "NT-Xent",
        "code": "nt_xent",
        "pretrain_history": EXP1_BASE_DIR / "loss_nt_xent" / "pretrain" / "curves" / "pretraining_history.csv",
        "probe_history": EXP1_BASE_DIR / "loss_nt_xent" / "linear_probe_label10" / "curves" / "training_history.csv",
        "test_metrics": EXP1_BASE_DIR / "loss_nt_xent" / "final_test_label10" / "results" / "test_metrics.json",
        "test_matrix": EXP1_BASE_DIR / "loss_nt_xent" / "final_test_label10" / "matrices" / "test_confusion_matrix.png",
    },
    {
        "name": "Triplet",
        "code": "triplet",
        "pretrain_history": EXP1_BASE_DIR / "loss_triplet" / "pretrain" / "curves" / "pretraining_history.csv",
        "probe_history": EXP1_BASE_DIR / "loss_triplet" / "linear_probe_label10" / "curves" / "training_history.csv",
        "test_metrics": EXP1_BASE_DIR / "loss_triplet" / "final_test_label10" / "results" / "test_metrics.json",
        "test_matrix": EXP1_BASE_DIR / "loss_triplet" / "final_test_label10" / "matrices" / "test_confusion_matrix.png",
    },
    {
        "name": "Contrastive",
        "code": "contrastive",
        "pretrain_history": EXP1_BASE_DIR / "loss_contrastive" / "pretrain" / "curves" / "pretraining_history.csv",
        "probe_history": EXP1_BASE_DIR / "loss_contrastive" / "linear_probe_label10" / "curves" / "training_history.csv",
        "test_metrics": EXP1_BASE_DIR / "loss_contrastive" / "final_test_label10" / "results" / "test_metrics.json",
        "test_matrix": EXP1_BASE_DIR / "loss_contrastive" / "final_test_label10" / "matrices" / "test_confusion_matrix.png",
    },
]

EXP2_BASE_DIR = OUTPUT_DIR / "additional_experiment_2" / "additional_experiment_2_resnet18_simclr_v1"
EXP2_HEAD_CONFIGS = [
    {
        "name": "MLP",
        "code": "mlp",
        "pretrain_history": EXP2_BASE_DIR / "head_mlp" / "pretrain" / "curves" / "pretraining_history.csv",
        "probe_history": EXP2_BASE_DIR / "head_mlp" / "linear_probe_label10" / "curves" / "training_history.csv",
        "test_metrics": EXP2_BASE_DIR / "head_mlp" / "final_test_label10" / "results" / "test_metrics.json",
        "test_matrix": EXP2_BASE_DIR / "head_mlp" / "final_test_label10" / "matrices" / "test_confusion_matrix.png",
    },
    {
        "name": "MLP+BN",
        "code": "mlp_bn",
        "pretrain_history": EXP2_BASE_DIR / "head_mlp_bn" / "pretrain" / "curves" / "pretraining_history.csv",
        "probe_history": EXP2_BASE_DIR / "head_mlp_bn" / "linear_probe_label10" / "curves" / "training_history.csv",
        "test_metrics": EXP2_BASE_DIR / "head_mlp_bn" / "final_test_label10" / "results" / "test_metrics.json",
        "test_matrix": EXP2_BASE_DIR / "head_mlp_bn" / "final_test_label10" / "matrices" / "test_confusion_matrix.png",
    },
    {
        "name": "MLP-Wide",
        "code": "mlp_wide",
        "pretrain_history": EXP2_BASE_DIR / "head_mlp_wide" / "pretrain" / "curves" / "pretraining_history.csv",
        "probe_history": EXP2_BASE_DIR / "head_mlp_wide" / "linear_probe_label10" / "curves" / "training_history.csv",
        "test_metrics": EXP2_BASE_DIR / "head_mlp_wide" / "final_test_label10" / "results" / "test_metrics.json",
        "test_matrix": EXP2_BASE_DIR / "head_mlp_wide" / "final_test_label10" / "matrices" / "test_confusion_matrix.png",
    },
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def copy_image(source, target_name):
    source = Path(source)
    if not source.exists():
        print(f"Skip missing image: {source}")
        return None
    target = RAW_DIR / target_name
    shutil.copy2(source, target)
    print(f"Copied: {target}")
    return target


def plot_main_results():
    groups = []
    baseline_acc = []
    probe_acc = []
    baseline_f1 = []
    probe_f1 = []

    for config in CONFIGS:
        if not config["baseline_test"].exists() or not config["simclr_test"].exists():
            print(f"Skip missing test metrics for {config['code']}")
            continue
        baseline_metrics = read_json(config["baseline_test"])
        simclr_metrics = read_json(config["simclr_test"])
        groups.append(config["name"])
        baseline_acc.append(baseline_metrics["test_accuracy"])
        probe_acc.append(simclr_metrics["test_accuracy"])
        baseline_f1.append(baseline_metrics["test_f1"])
        probe_f1.append(simclr_metrics["test_f1"])

    if not groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.35
    positions = list(range(len(groups)))

    axes[0].bar([p - width / 2 for p in positions], baseline_acc, width=width, label="baseline")
    axes[0].bar([p + width / 2 for p in positions], probe_acc, width=width, label="linear_probe")
    axes[0].set_title("Test Accuracy Comparison")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(groups, rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    axes[1].bar([p - width / 2 for p in positions], baseline_f1, width=width, label="baseline")
    axes[1].bar([p + width / 2 for p in positions], probe_f1, width=width, label="linear_probe")
    axes[1].set_title("Test F1 Comparison")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(groups, rotation=20)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "main_results_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_pretraining_loss_compare():
    series = {
        "simclr_v1": OUTPUT_DIR / "simclr_pretrain" / "simclr_pretrain_resnet18_simclr_v1" / "curves" / "pretraining_history.csv",
        "simclr_v2": OUTPUT_DIR / "simclr_pretrain" / "simclr_pretrain_resnet18_simclr_v2" / "curves" / "pretraining_history.csv",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for label, path in series.items():
        if not path.exists():
            print(f"Skip missing pretraining csv: {path}")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        losses = [float(row["train_loss"]) for row in rows]
        ax.plot(epochs, losses, marker="o", label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("ResNet18 Pretraining Loss Under Different Augmentations")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.legend()
    fig.tight_layout()
    output_path = GENERATED_DIR / "pretraining_loss_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_curve_compare():
    series = {
        "baseline_val_acc": OUTPUT_DIR / "train_val" / "baseline_resnet18_label10_val20" / "curves" / "training_history.csv",
        "linear_probe_val_acc": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label10_val20" / "curves" / "training_history.csv",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plotted = False
    for label, path in series.items():
        if not path.exists():
            print(f"Skip missing training csv: {path}")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        if "baseline" in label:
            axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="baseline_train_loss")
            axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="baseline_val_loss")
            axes[1].plot(epochs, [float(row["val_acc"]) for row in rows], label="baseline_val_acc")
        else:
            axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="linear_probe_train_loss")
            axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="linear_probe_val_loss")
            axes[1].plot(epochs, [float(row["val_acc"]) for row in rows], label="linear_probe_val_acc")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_title("ResNet18 Label10 Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("ResNet18 Label10 Val Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "curve_compare_resnet18_label10.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_linear_probe_augmentation_compare():
    series = {
        "simclr_v1": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label10_val20" / "curves" / "training_history.csv",
        "simclr_v2": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v2_label10_val20" / "curves" / "training_history.csv",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False
    for label, path in series.items():
        if not path.exists():
            print(f"Skip missing linear probe csv: {path}")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label=label)
        axes[0, 1].plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=label)
        axes[1, 0].plot(epochs, [float(row["val_acc"]) for row in rows], marker="o", label=label)
        axes[1, 1].plot(epochs, [float(row["val_f1"]) for row in rows], marker="o", label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0, 0].set_title("Linear Probe Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("Linear Probe Val Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("Linear Probe Val Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    axes[1, 1].set_title("Linear Probe Val F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "linear_probe_augmentation_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_method_comparison_by_setting():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    plotted = False

    for col, config in enumerate(CONFIGS):
        if not config["baseline_history"].exists() or not config["simclr_history"].exists():
            print(f"Skip missing history for {config['code']}")
            continue
        baseline_rows = read_csv(config["baseline_history"])
        simclr_rows = read_csv(config["simclr_history"])
        baseline_epochs = [int(row["epoch"]) for row in baseline_rows]
        simclr_epochs = [int(row["epoch"]) for row in simclr_rows]

        axes[0, col].plot(baseline_epochs, [float(row["val_loss"]) for row in baseline_rows], marker="o", label="baseline")
        axes[0, col].plot(simclr_epochs, [float(row["val_loss"]) for row in simclr_rows], marker="o", label="simclr")
        axes[0, col].set_title(f"{config['code']} Val Loss")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Loss")
        axes[0, col].legend()

        axes[1, col].plot(baseline_epochs, [float(row["val_acc"]) for row in baseline_rows], marker="o", label="baseline")
        axes[1, col].plot(simclr_epochs, [float(row["val_acc"]) for row in simclr_rows], marker="o", label="simclr")
        axes[1, col].set_title(f"{config['code']} Val Accuracy")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Accuracy")
        axes[1, col].set_ylim(0.0, 1.0)
        axes[1, col].legend()
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    fig.tight_layout()
    output_path = GENERATED_DIR / "method_comparison_by_setting.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_method_test_comparison_by_setting():
    labels = []
    baseline_acc = []
    simclr_acc = []
    baseline_f1 = []
    simclr_f1 = []

    for config in CONFIGS:
        if not config["baseline_test"].exists() or not config["simclr_test"].exists():
            print(f"Skip missing test metrics for {config['code']}")
            continue
        baseline_metrics = read_json(config["baseline_test"])
        simclr_metrics = read_json(config["simclr_test"])
        labels.append(config["code"])
        baseline_acc.append(baseline_metrics["test_accuracy"])
        simclr_acc.append(simclr_metrics["test_accuracy"])
        baseline_f1.append(baseline_metrics["test_f1"])
        simclr_f1.append(simclr_metrics["test_f1"])

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35
    positions = list(range(len(labels)))

    axes[0].bar([p - width / 2 for p in positions], baseline_acc, width=width, label="baseline")
    axes[0].bar([p + width / 2 for p in positions], simclr_acc, width=width, label="simclr")
    axes[0].set_title("Baseline vs SimCLR Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    axes[1].bar([p - width / 2 for p in positions], baseline_f1, width=width, label="baseline")
    axes[1].bar([p + width / 2 for p in positions], simclr_f1, width=width, label="simclr")
    axes[1].set_title("Baseline vs SimCLR Test F1")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "method_test_comparison_by_setting.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_simclr_training_across_settings():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False

    for config in CONFIGS:
        path = config["simclr_history"]
        if not path.exists():
            print(f"Skip missing simclr history for {config['code']}")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        label = config["code"]
        axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label=label)
        axes[0, 1].plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=label)
        axes[1, 0].plot(epochs, [float(row["val_acc"]) for row in rows], marker="o", label=label)
        axes[1, 1].plot(epochs, [float(row["val_f1"]) for row in rows], marker="o", label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0, 0].set_title("SimCLR Train Loss Across Settings")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("SimCLR Val Loss Across Settings")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("SimCLR Val Accuracy Across Settings")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    axes[1, 1].set_title("SimCLR Val F1 Across Settings")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "simclr_training_across_settings.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_simclr_test_across_settings():
    labels = []
    acc = []
    f1 = []

    for config in CONFIGS:
        path = config["simclr_test"]
        if not path.exists():
            print(f"Skip missing simclr test metrics for {config['code']}")
            continue
        metrics = read_json(path)
        labels.append(config["code"])
        acc.append(metrics["test_accuracy"])
        f1.append(metrics["test_f1"])

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = list(range(len(labels)))

    axes[0].bar(positions, acc, width=0.6)
    axes[0].set_title("SimCLR Test Accuracy Across Settings")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(positions, f1, width=0.6)
    axes[1].set_title("SimCLR Test F1 Across Settings")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    output_path = GENERATED_DIR / "simclr_test_across_settings.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp1_pretrain_compare():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    plotted = False

    for ax, config in zip(axes, EXP1_LOSS_CONFIGS):
        path = config["pretrain_history"]
        if not path.exists():
            print(f"Skip missing additional exp1 pretrain csv: {path}")
            ax.axis("off")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        losses = [float(row["train_loss"]) for row in rows]
        ax.plot(epochs, losses, marker="o")
        ax.set_title(config["name"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    fig.suptitle("Additional Experiment 1: Pretraining Curves of Different Contrastive Losses", fontsize=12)
    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp1_pretrain_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp1_probe_compare():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False

    for config in EXP1_LOSS_CONFIGS:
        path = config["probe_history"]
        if not path.exists():
            print(f"Skip missing additional exp1 probe csv: {path}")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        label = config["name"]
        axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label=label)
        axes[0, 1].plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=label)
        axes[1, 0].plot(epochs, [float(row["val_acc"]) for row in rows], marker="o", label=label)
        axes[1, 1].plot(epochs, [float(row["val_f1"]) for row in rows], marker="o", label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0, 0].set_title("Linear Probe Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("Linear Probe Val Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("Linear Probe Val Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    axes[1, 1].set_title("Linear Probe Val F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp1_probe_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp1_test_compare():
    labels = []
    acc = []
    f1 = []

    for config in EXP1_LOSS_CONFIGS:
        path = config["test_metrics"]
        if not path.exists():
            print(f"Skip missing additional exp1 test metrics: {path}")
            continue
        metrics = read_json(path)
        labels.append(config["name"])
        acc.append(metrics["test_accuracy"])
        f1.append(metrics["test_f1"])

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = list(range(len(labels)))

    axes[0].bar(positions, acc, width=0.6)
    axes[0].set_title("Additional Experiment 1 Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(positions, f1, width=0.6)
    axes[1].set_title("Additional Experiment 1 Test F1")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp1_test_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp1_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plotted = False

    for ax, config in zip(axes, EXP1_LOSS_CONFIGS):
        path = config["test_matrix"]
        if not path.exists():
            print(f"Skip missing additional exp1 confusion matrix: {path}")
            ax.axis("off")
            continue
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(config["name"])
        ax.axis("off")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp1_confusion_matrices.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp2_pretrain_compare():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    plotted = False

    for ax, config in zip(axes, EXP2_HEAD_CONFIGS):
        path = config["pretrain_history"]
        if not path.exists():
            print(f"Skip missing additional exp2 pretrain csv: {path}")
            ax.axis("off")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        losses = [float(row["train_loss"]) for row in rows]
        ax.plot(epochs, losses, marker="o")
        ax.set_title(config["name"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    fig.suptitle("Additional Experiment 2: Pretraining Curves of Different Projection Heads", fontsize=12)
    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp2_pretrain_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp2_probe_compare():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False

    for config in EXP2_HEAD_CONFIGS:
        path = config["probe_history"]
        if not path.exists():
            print(f"Skip missing additional exp2 probe csv: {path}")
            continue
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        label = config["name"]
        axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label=label)
        axes[0, 1].plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=label)
        axes[1, 0].plot(epochs, [float(row["val_acc"]) for row in rows], marker="o", label=label)
        axes[1, 1].plot(epochs, [float(row["val_f1"]) for row in rows], marker="o", label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0, 0].set_title("Linear Probe Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("Linear Probe Val Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("Linear Probe Val Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    axes[1, 1].set_title("Linear Probe Val F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp2_probe_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp2_test_compare():
    labels = []
    acc = []
    f1 = []

    for config in EXP2_HEAD_CONFIGS:
        path = config["test_metrics"]
        if not path.exists():
            print(f"Skip missing additional exp2 test metrics: {path}")
            continue
        metrics = read_json(path)
        labels.append(config["name"])
        acc.append(metrics["test_accuracy"])
        f1.append(metrics["test_f1"])

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = list(range(len(labels)))

    axes[0].bar(positions, acc, width=0.6)
    axes[0].set_title("Additional Experiment 2 Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(positions, f1, width=0.6)
    axes[1].set_title("Additional Experiment 2 Test F1")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp2_test_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_additional_exp2_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plotted = False

    for ax, config in zip(axes, EXP2_HEAD_CONFIGS):
        path = config["test_matrix"]
        if not path.exists():
            print(f"Skip missing additional exp2 confusion matrix: {path}")
            ax.axis("off")
            continue
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(config["name"])
        ax.axis("off")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    fig.tight_layout()
    output_path = GENERATED_DIR / "additional_exp2_confusion_matrices.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def copy_raw_images():
    images = {
        "augmentation_simclr_v1.png": OUTPUT_DIR / "augmentation_views" / "views_simclr_v1-simclr_v2_img4_pair2" / "simclr_v1" / "augmentation_views.png",
        "augmentation_simclr_v2.png": OUTPUT_DIR / "augmentation_views" / "views_simclr_v1-simclr_v2_img4_pair2" / "simclr_v2" / "augmentation_views.png",
        "baseline_resnet18_label10_curves.png": OUTPUT_DIR / "train_val" / "baseline_resnet18_label10_val20" / "curves" / "training_curves.png",
        "linear_probe_resnet18_label10_curves.png": OUTPUT_DIR / "linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label10_val20" / "curves" / "training_curves.png",
        "baseline_resnet18_label10_cm.png": OUTPUT_DIR / "final_test" / "baseline_resnet18_label10_val20" / "matrices" / "test_confusion_matrix.png",
        "linear_probe_resnet18_label10_cm.png": OUTPUT_DIR / "final_test_linear_probe" / "linear_probe_simclr_pretrain_resnet18_simclr_v1_label10_val20" / "matrices" / "test_confusion_matrix.png",
        "baseline_mobilenet_label10_cm.png": OUTPUT_DIR / "final_test" / "baseline_mobilenet_v2_label10_val20" / "matrices" / "test_confusion_matrix.png",
        "linear_probe_mobilenet_label10_cm.png": OUTPUT_DIR / "final_test_linear_probe" / "linear_probe_simclr_pretrain_mobilenet_v2_simclr_v1_label10_val20" / "matrices" / "test_confusion_matrix.png",
    }
    for target_name, source in images.items():
        copy_image(source, target_name)


def main():
    ensure_dirs()
    copy_raw_images()
    plot_main_results()
    plot_pretraining_loss_compare()
    plot_curve_compare()
    plot_linear_probe_augmentation_compare()
    plot_method_comparison_by_setting()
    plot_method_test_comparison_by_setting()
    plot_simclr_training_across_settings()
    plot_simclr_test_across_settings()
    plot_additional_exp1_pretrain_compare()
    plot_additional_exp1_probe_compare()
    plot_additional_exp1_test_compare()
    plot_additional_exp1_confusion_matrices()
    plot_additional_exp2_pretrain_compare()
    plot_additional_exp2_probe_compare()
    plot_additional_exp2_test_compare()
    plot_additional_exp2_confusion_matrices()


if __name__ == "__main__":
    main()
