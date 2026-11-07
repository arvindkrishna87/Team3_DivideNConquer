"""Per-update and per-epoch train/validation loss, one figure per fold."""
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE = "/home/cny2467/Presentation2_base_code"
OUT_DIR = os.path.join(BASE, "run_result_analysis", "kimia")
RUNS = os.path.join(BASE, "MIL_training", "runs", "kimia")

FOLDS = {
    1: "run_20261106_111329",
    2: "run_20261106_111638",
    3: "run_20261106_112513",
    4: "run_20261106_113022",
    5: "run_20261106_113620",
}


def plot_fold(df: pd.DataFrame, fold: int, out_path: str) -> None:
    epoch = (
        df.groupby("epoch", as_index=False)
        .agg(train_loss=("train_loss", "mean"), val_loss=("val_loss", "last"))
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    axes[0].plot(df["step"], df["train_loss"], label="Train loss", linewidth=1.0)
    axes[0].plot(df["step"], df["val_loss"], label="Validation loss", linewidth=1.0)
    axes[0].set_title(f"KimiaNet, fold {fold}: loss per gradient update")
    axes[0].set_xlabel("Gradient update")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epoch["epoch"], epoch["train_loss"], marker="o", label="Train loss", linewidth=1.2)
    axes[1].plot(epoch["epoch"], epoch["val_loss"], marker="o", label="Validation loss", linewidth=1.2)
    axes[1].set_title(f"KimiaNet, fold {fold}: loss per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(epoch["epoch"])
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for fold, run_name in FOLDS.items():
        path = os.path.join(RUNS, run_name, "step_losses.csv")
        df = pd.read_csv(path)
        out = os.path.join(OUT_DIR, f"fold{fold:02d}_loss.png")
        plot_fold(df, fold, out)
        print(f"fold {fold}: {len(df)} steps, {df['epoch'].nunique()} epochs -> {out}")


if __name__ == "__main__":
    main()
