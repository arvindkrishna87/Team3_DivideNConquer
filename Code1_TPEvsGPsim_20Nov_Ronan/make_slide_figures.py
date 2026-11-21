#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------------------------------------------
CODED WITH CLAUDE OPUS 5.5
--------------------------------------------------------------------------------------------------------------------------------

Reads ROOT/<setting>_init16/summary.json and ROOT/fixed_init16/evaluated_x.json and writes:
    fig1_problem.png       objective, the 16 initial trials and the success band
    fig2_tpe_spread.png    where TPE's first 16 new trials land vs random search (1-D, 16 initial)
    fig3_success.png       success curves in 1-D, 6-D, 10-D (16 initial): TPE, GP-EI, GP-EI with xi, Random
    fig4_noisy.png         success curves under noise (16 initial): all GP-EI variants, TPE, Random

 figures are written to results/slide_figures/
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13})

COLORS = {"TPE": "#4E2A84", "GP-EI": "#D85A30", "GP-EI-xi": "#1D9E75", "GP-EI-latent": "#3B8BD4",
          "GP-EI-minobs": "#BA7517", "Random": "#888780"}
LABELS = {"TPE": "TPE", "GP-EI": "GP-EI (ξ = 0)", "GP-EI-xi": "GP-EI (ξ = 0.01)",
          "GP-EI-latent": "GP-EI, noise-free σ", "GP-EI-minobs": "GP-EI, b = lowest observation",
          "Random": "Random search"}
STYLES = {"Random": "--"}
LOCAL_BAND = (2.0, 4.0)


def f1(x):
    x = np.asarray(x, dtype=float)
    return 0.65 - 0.30 * np.exp(-(x - 3) ** 2 / (2 * 0.7 ** 2)) - 0.60 * np.exp(-(x - 8) ** 2 / (2 * 0.4 ** 2))


def load(root, setting):
    with open(os.path.join(root, f"{setting}_init16", "summary.json")) as fh:
        d = json.load(fh)
    return d["summary"][setting], d["args"]


def success_axes(ax, summ, args, methods, title):
    n0 = args["n_init"]
    for m in methods:
        curve = np.array(summ[m]["success_by_trial"])[n0 - 1:]          # from the last initial trial on
        ax.plot(np.arange(len(curve)), curve, color=COLORS[m], lw=3, ls=STYLES.get(m, "-"), label=LABELS[m])
    ax.set_title(title)
    ax.set_xlabel("new trials")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, args["n_new"])
    ax.grid(alpha=0.3)


def main(root):
    out = os.path.join(root, "slide_figures")
    os.makedirs(out, exist_ok=True)
    x = np.linspace(0, 10, 2001)

    # 1. The problem
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhspan(0.0, 0.10, xmin=0, xmax=1, color="#E1F5EE", zorder=0)
    ax.plot(x, f1(x), color="black", lw=2.5)
    i16 = np.linspace(2.0, 3.9, 16)
    ax.scatter(i16, f1(i16), s=55, color="#4E2A84", zorder=5, label="16 initial trials (all methods)")
    ax.annotate("local minimum\nf ≈ 0.35", xy=(3, 0.35), xytext=(4.2, 0.18), arrowprops={"arrowstyle": "->"})
    ax.annotate("global minimum\nf* = 0.05", xy=(8, 0.05), xytext=(8.65, 0.22), arrowprops={"arrowstyle": "->"})
    ax.text(0.1, 0.03, "success: f ≤ f* + 0.05", ha="left", fontsize=14, color="#085041")
    ax.set_xlabel("hyperparameter x₁")
    ax.set_ylabel("validation loss f(x)")
    ax.set_ylim(-0.02, 0.9)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_problem.png"), dpi=200)
    plt.close(fig)

    # 2. Where TPE's first 16 new trials land (1-D, 16 initial)
    with open(os.path.join(root, "fixed_init16", "evaluated_x.json")) as fh:
        ev = json.load(fh)["fixed"]
    n0 = 16
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(0, 10, 41)
    ax.axvspan(*LOCAL_BAND, color="#EEEDFE", zorder=0)
    for m in ("TPE", "Random"):
        new = np.array([run[n0:n0 + 16] for run in ev[m]]).ravel()
        share = np.mean((new >= LOCAL_BAND[0]) & (new <= LOCAL_BAND[1]))
        ax.hist(new, bins=bins, density=True, histtype="step", lw=3, color=COLORS[m], ls=STYLES.get(m, "-"),
                label=f"{LABELS[m]}: {share:.0%} within 2 ≤ x₁ ≤ 4")
    ax.text(3.0, ax.get_ylim()[1] * 0.92, "16 initial\ntrials here", ha="center", va="top", fontsize=13,
            color="#3C3489")
    ax.set_xlabel("hyperparameter x₁")
    ax.set_ylabel("density of first 16 new trials\n(100 runs)")
    ax.set_title("TPE moves away from its 16 initial trials (1-D, noise-free)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_tpe_spread.png"), dpi=200)
    plt.close(fig)

    # 3. Success: 1-D, 6-D, 10-D
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    methods = ["GP-EI-xi", "GP-EI", "TPE", "Random"]
    for ax, (s, title) in zip(axes, (("fixed", "1-D"), ("fixed6d", "6-D"), ("fixed10d", "10-D"))):
        summ, args = load(root, s)
        success_axes(ax, summ, args, methods, title)
    axes[0].set_ylabel("P(found global minimum)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(out, "fig3_success.png"), dpi=200)
    plt.close(fig)

    # 4. Noise
    summ, args = load(root, "noisy")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    success_axes(ax, summ, args, ["GP-EI-minobs", "GP-EI-latent", "GP-EI-xi", "GP-EI", "TPE", "Random"],
                 "1-D with observation noise (sd 0.03)")
    ax.set_ylabel("P(found global minimum)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_noisy.png"), dpi=200)
    plt.close(fig)
    print(f"Saved 4 figures to {os.path.abspath(out)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
