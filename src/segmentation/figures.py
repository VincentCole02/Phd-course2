"""Render the recorded metrics and representative held-out test predictions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


INK = "#172934"
MUTED = "#526873"
BLUE = "#146a85"
GREEN = "#26846b"
PALE = "#d7e5e9"


def setup_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(PALE)
    axis.tick_params(color=PALE, labelcolor=MUTED)
    axis.grid(axis="y", color=PALE, alpha=0.6)
    axis.set_axisbelow(True)


def training_figure(path, metrics, history):
    fig = plt.figure(figsize=(10.7, 6.3), facecolor="white")
    fig.text(0.09, 0.93, "EGE-UNet on ISIC2017", fontsize=24, color=INK, weight="bold")
    fig.text(
        0.09, 0.879,
        f"{metrics['epochs']} epochs  /  {metrics['splits']['train']:,} train  /  "
        f"{metrics['splits']['val']} validation  /  {metrics['splits']['test']} test images",
        fontsize=11, color=MUTED,
    )
    for x, key, label in [(0.09, "dice", "Test Dice"), (0.36, "iou", "Test IoU")]:
        fig.text(x, 0.775, label, fontsize=11, color=MUTED)
        fig.text(x, 0.711, f"{metrics['test'][key]:.2%}", fontsize=25, color=BLUE, weight="bold")
        fig.text(x, 0.664, f"Paper: {metrics['paper_reference'][key]:.2%}", fontsize=10, color=MUTED)
    fig.text(0.63, 0.775, "Selected checkpoint", fontsize=11, color=MUTED)
    fig.text(0.63, 0.711, f"Epoch {metrics['best_epoch']}", fontsize=25, color=INK, weight="bold")
    fig.text(0.63, 0.664, "Highest validation Dice", fontsize=10, color=MUTED)

    loss_ax = fig.add_axes((0.09, 0.16, 0.37, 0.39))
    val_ax = fig.add_axes((0.57, 0.16, 0.37, 0.39))
    epochs = [row["epoch"] for row in history]
    loss_ax.plot(epochs, [row["train_loss"] for row in history], color=BLUE, lw=2)
    loss_ax.set_title("Training loss", loc="left", fontsize=12, color=INK, weight="bold")
    val_ax.plot(epochs, [row["val_dice"] * 100 for row in history], color=BLUE, lw=2, label="Dice")
    val_ax.plot(epochs, [row["val_iou"] * 100 for row in history], color=GREEN, lw=2, label="IoU")
    val_ax.axvline(metrics["best_epoch"], color=MUTED, linestyle=":", linewidth=1)
    val_ax.set_title("Validation overlap (%)", loc="left", fontsize=12, color=INK, weight="bold")
    val_ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=9)
    for axis in (loss_ax, val_ax):
        setup_axis(axis)
        axis.set_xlabel("Epoch", color=MUTED)
    fig.text(
        0.09, 0.049, "Paper reference uses 300 epochs and its own random 70/30 split.",
        fontsize=9, color=MUTED,
    )
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)


def prediction_figure(path, dataset, per_image, predict):
    ordered = sorted(enumerate(per_image), key=lambda item: (item[1]["dice"], item[1]["isic_id"]))
    indices = sorted({round((len(ordered) - 1) * q) for q in (0, 0.10, 0.25, 0.50, 0.75, 0.95)})
    fig, axes = plt.subplots(4, len(indices), figsize=(12.2, 9.3), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.977, top=0.85, bottom=0.091, wspace=0.13, hspace=0.09)
    fig.text(0.045, 0.937, "Held-out ISIC2017 predictions", fontsize=22, color=INK, weight="bold")
    fig.text(
        0.045, 0.9, "Examples from the worst case through the upper Dice percentiles.",
        fontsize=11, color=MUTED,
    )
    for row, label in enumerate(("Image", "Reference\nmask", "Predicted\nmask", "Error\noverlay")):
        box = axes[row, 0].get_position()
        fig.text(0.044, box.y0 + box.height / 2, label, fontsize=10, color=INK, va="center")
    for col, rank in enumerate(indices):
        index, row = ordered[rank]
        image, target, isic_id = dataset.raw(index)
        assert isic_id == row["isic_id"]
        prediction = predict(image)
        overlay = image.astype(np.float32) / 255 * 0.55
        for region, color in (
            (prediction & target, [0.24, 0.78, 0.67]),
            (prediction & ~target, [1.0, 0.45, 0.19]),
            (~prediction & target, [0.36, 0.57, 1.0]),
        ):
            overlay[region] = 0.27 * overlay[region] + 0.73 * np.asarray(color)
        views = (image, target, prediction, overlay)
        for axis, view in zip(axes[:, col], views):
            axis.imshow(view, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            axis.set_axis_off()
        axes[0, col].set_title(f"{isic_id}\nDice {row['dice']:.1%}", fontsize=9, color=INK, pad=8)

    for x, label, color in (
        (0.13, "Overlap", "#288c79"), (0.32, "False positive", "#c46128"),
        (0.56, "False negative", "#3565bd"),
    ):
        fig.text(x, 0.045, f"●  {label}", fontsize=10, color=color)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def save_figures(output, metrics, history, per_image, dataset, predict):
    if history is not None:
        training_figure(output / "training-results.png", metrics, history)
    prediction_figure(output / "test-predictions.png", dataset, per_image, predict)
