import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LOGS = "/home/vnc0125/presentation_3/taylor_logs/exp99_*.out"
OUT = "/home/vnc0125/presentation_3/taylor_slide_graphics_data"
ORDER = ["baseline", "stain0.3", "slice0.3", "emb64", "emb128", "emb256"]
NAMES = {"baseline": "Baseline", "stain0.3": "Stain dropout 0.3", "slice0.3": "Slice dropout 0.3",
         "emb64": "Embedding 64", "emb128": "Embedding 128", "emb256": "Embedding 256"}
os.makedirs(OUT, exist_ok=True)

# find each finished run from its log (newest log wins if a task was rerun)
runs = {}
for lf in sorted(glob.glob(LOGS), key=os.path.getmtime):
    txt = open(lf, errors="ignore").read()
    m = re.search(r"Task (\d+) \| tag=(\S+) \| split=(\S+)", txt)
    d = re.search(r"Created run directory: (\S+)", txt)
    if m and d:
        runs[(m.group(2), int(m.group(3)[-6:-4]))] = d.group(1)
hist = {}
for (tag, sp), rd in runs.items():
    if os.path.isfile(os.path.dirname(rd) + "/FINISHED") and os.path.isfile(rd + "/history.json"):
        hist[(tag, sp)] = json.load(open(rd + "/history.json"))

def make_grid(tags, fname):
    tags = [t for t in tags if any((t, s) in hist for s in range(1, 6))]
    fig, axes = plt.subplots(len(tags), 5, figsize=(13, 2.1 * len(tags) + 0.8), sharey=True, squeeze=False)
    for i, t in enumerate(tags):
        for j in range(5):
            ax = axes[i][j]; h = hist.get((t, j + 1))
            if h is None:
                ax.text(0.5, 0.5, "run missing", ha="center", va="center", transform=ax.transAxes, color="gray")
            else:
                ep = np.arange(1, len(h["val_losses"]) + 1)
                ax.plot(ep, h["train_clean_losses"], color="#2F6FBF", lw=1.8, label="Train (no dropout)")
                ax.plot(ep, h["val_losses"], color="#E08A1E", lw=1.8, label="Validation")
                ax.axvline(int(np.argmin(h["val_losses"])) + 1, color="gray", ls=":", lw=1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.tick_params(labelsize=7)
            if i == 0: ax.set_title("Split %d" % (j + 1), fontsize=10)
            if i == len(tags) - 1: ax.set_xlabel("epoch", fontsize=8)
            if j == 0: ax.set_ylabel(NAMES[t], fontsize=8, fontweight="bold")
    axes[0][4].legend(fontsize=7, loc="upper right")
    fig.suptitle("Training loss (eval mode) and validation loss per split; dotted line = best epoch", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT + "/" + fname, dpi=200); plt.close(fig)
    print("saved", fname, "with", len(tags), "rows")

make_grid(ORDER, "loss_grid_all.png")
make_grid(["baseline", "stain0.3", "slice0.3"], "loss_grid_dropout.png")
make_grid(["baseline", "emb64", "emb128", "emb256"], "loss_grid_embedding.png")
