"""One deviation plot per test fold."""
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE = "/home/cny2467/Presentation2_base_code"
OUT_DIR = os.path.join(BASE, "run_result_analysis", "kimia")

FOLDS = {
    1: "run_20261106_111329",
    2: "run_20261106_111638",
    3: "run_20261106_112513",
    4: "run_20261106_113022",
    5: "run_20261106_113620",
}


def plot_fold(data: pd.DataFrame, title: str, out_path: str) -> None:
    data = data.sort_values(["true_label", "case_id"]).reset_index(drop=True)
    n = len(data)
    fig_h = max(4.5, 0.38 * n)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for i, row in data.iterrows():
        true_x = float(row["true_label"])
        pred_x = float(row["prob_class1"])
        ax.plot([true_x, pred_x], [i, i], color="0.65", linewidth=1.0, zorder=1)
        ax.scatter(true_x, i, s=42, c="#2ca02c", zorder=3)
        ax.scatter(pred_x, i, s=42, c="black", marker="x", linewidths=1.3, zorder=4)

    ax.axvline(0.5, color="#2ca02c", linestyle="--", linewidth=1, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(data["case_id"].astype(str), fontsize=9)
    ax.set_ylim(-1, n)
    ax.set_xlim(-0.08, 1.08)
    ax.set_xlabel("True Label (0 or 1)  <->  Predicted Probability (High-grade)")
    ax.set_ylabel("Case (sorted by True Label)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE, "run_result_analysis", "uni2"), exist_ok=True)
    runs = os.path.join(BASE, "MIL_training", "runs", "kimia")

    for fold, run_name in FOLDS.items():
        path = os.path.join(runs, run_name, "predictions_best.csv")
        data = pd.read_csv(path)
        data["case_id"] = data["case_id"].astype(int)
        data["true_label"] = data["true_label"].astype(int)
        out = os.path.join(OUT_DIR, f"fold{fold:02d}_deviation.png")
        n = len(data)
        correct = data["true_label"] == data["predicted_label"].astype(int)
        print(f"fold {fold}: n={n} acc={correct.mean():.4f} -> {out}")
        plot_fold(data, f"Deviation Plot: True Label vs Predicted Probability\nKimiaNet, fold {fold}", out)


if __name__ == "__main__":
    main()
