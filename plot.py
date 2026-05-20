import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_history(history: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train_acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="val_acc", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], label="val_f1", linewidth=2)
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path: str | Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
