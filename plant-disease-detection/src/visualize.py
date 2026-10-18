"""Plots used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: dict, title: str, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    b = history.get("phase_boundary")
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(history[key], label="train")
        ax.plot(history["val_" + key], label="validation")
        if b:
            ax.axvline(b - 0.5, color="grey", ls="--", lw=1)
            ax.text(b - 0.4, ax.get_ylim()[0], " fine-tuning", va="bottom", fontsize=8, color="grey")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f"{title}: training curves")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, class_names, title: str, path, normalize: bool = True) -> None:
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * .32), max(7, n * .30)))
    im = ax.imshow(data, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [c.replace("___", ": ").replace("_", " ")[:34] for c in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6); ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}: confusion matrix" + (" (row-normalised)" if normalize else ""))
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_class_f1(report: dict, class_names, title: str, path) -> None:
    f1 = [report[c]["f1-score"] for c in class_names]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(8, max(6, len(class_names) * .22)))
    ax.barh([class_names[i].replace("___", ": ").replace("_", " ") for i in order],
            [f1[i] for i in order], color="#2E7D4F")
    ax.set_xlim(0, 1.0); ax.set_xlabel("F1-score"); ax.tick_params(axis="y", labelsize=6)
    ax.set_title(f"{title}: per-class F1"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_comparison(rows: dict, path) -> None:
    """rows = {model: {metric: value}}. Four small bar charts."""
    names = list(rows)
    panels = [("accuracy", "Test accuracy"), ("f1_macro", "Macro F1"),
              ("params_millions", "Parameters (millions)"), ("inference_ms", "Inference (ms / image)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    colors = ["#2E7D4F", "#B5471B", "#3B6E9E", "#8A6D1D"]
    for ax, (k, t) in zip(axes, panels):
        vals = [rows[n][k] for n in names]
        ax.bar(names, vals, color=colors[:len(names)])
        ax.set_title(t, fontsize=10); ax.grid(axis="y", alpha=.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        if k in ("accuracy", "f1_macro"):
            ax.set_ylim(min(vals) * 0.9 if min(vals) > 0.5 else 0, 1.02)
    fig.suptitle("MobileNetV2 vs ResNet50 on the PlantVillage test set")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
