import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import binomtest

LOGS = "/home/vnc0125/presentation_3/taylor_logs/exp99_*.out"
OUT = "/home/vnc0125/presentation_3/taylor_slide_graphics_data"
TAU = 0.05
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
        runs[(m.group(2), m.group(3))] = dict(tag=m.group(2), split=m.group(3)[-6:-4], run_dir=d.group(1))
good = {t: [] for t in ORDER}
for r in runs.values():
    parent = os.path.dirname(r["run_dir"])
    pf, hf = r["run_dir"] + "/predictions_best.csv", r["run_dir"] + "/history.json"
    if os.path.isfile(parent + "/FINISHED") and os.path.isfile(pf) and os.path.isfile(hf):
        good.setdefault(r["tag"], []).append(dict(r, pred=pd.read_csv(pf), hist=json.load(open(hf))))
for t in ORDER:
    print("%-9s finished splits: %d / 5" % (t, len(good.get(t, []))))

# results 
pooled = {}
for t in ORDER:
    if len(good.get(t, [])) == 5:
        p = pd.concat([g["pred"] for g in good[t]]).set_index("case_id")
        assert p.index.is_unique, t + ": a case appears in more than one test split"
        pooled[t] = p
rows = []
for t in ORDER:
    if t not in pooled:
        continue
    p = pooled[t]; ok = (p.true_label == p.predicted_label)
    row = {"Setting": NAMES[t], "Cases": len(p), "Accuracy": round(ok.mean(), 3),
           "Benign recall": round((p[p.true_label == 0].predicted_label == 0).mean(), 3),
           "High-grade recall": round((p[p.true_label == 1].predicted_label == 1).mean(), 3),
           "Right only for variant / baseline": "", "McNemar p": ""}
    if t != "baseline" and "baseline" in pooled:
        b = pooled["baseline"]
        okb = (b.true_label == b.predicted_label).reindex(p.index)
        only_v, only_b = int((ok & ~okb).sum()), int((~ok & okb).sum())
        row["Right only for variant / baseline"] = "%d / %d" % (only_v, only_b)
        row["McNemar p"] = "1.000" if only_v + only_b == 0 else "%.3f" % binomtest(min(only_v, only_b), only_v + only_b, 0.5).pvalue
    rows.append(row)
res = pd.DataFrame(rows)
print("\n=== SLIDE 9 TABLE ===\n" + res.to_string(index=False))
res.to_csv(OUT + "/results_table.csv", index=False)

# memorization
mrows = []
for t in ORDER:
    gs = good.get(t, [])
    if not gs:
        continue
    taus, gaps, bests, eps = [], [], [], []
    for g in gs:
        val, clean = np.array(g["hist"]["val_losses"]), np.array(g["hist"]["train_clean_losses"])
        b = int(np.argmin(val)); below = np.where(clean < TAU)[0]
        if len(below): taus.append(int(below[0]) + 1)
        gaps.append(val[b] - clean[b]); bests.append(b + 1); eps.append(len(val))
    mrows.append({"Setting": NAMES[t], "Splits": len(gs),
                  "Runs reaching clean loss < 0.05": "%d / %d" % (len(taus), len(gs)),
                  "Median epoch below 0.05": float(np.median(taus)) if taus else float("nan"),
                  "Mean val-train gap at best epoch": round(float(np.mean(gaps)), 3),
                  "Mean best epoch": round(float(np.mean(bests)), 1), "Mean epochs run": round(float(np.mean(eps)), 1)})
mem = pd.DataFrame(mrows)
print("\n=== SLIDE 10 TABLE ===\n" + mem.to_string(index=False))
mem.to_csv(OUT + "/memorization_table.csv", index=False)

# clean train loss (log scale) and validation loss, averaged over the 5 splits
def mean_curve(gs, key):
    cs = [list(g["hist"][key]) for g in gs]; n = max(len(c) for c in cs)
    return np.mean([c + [c[-1]] * (n - len(c)) for c in cs], axis=0)
colors = {"baseline": "#000000", "stain0.3": "#1F4E8C", "slice0.3": "#2F6FBF", "emb64": "#9ECAE1", "emb128": "#E08A1E", "emb256": "#C0392B"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for t in ORDER:
    gs = good.get(t, [])
    if len(gs) < 1:
        continue
    lw = 2.8 if t == "baseline" else 1.8
    for a, key in zip(ax, ["train_clean_losses", "val_losses"]):
        c = mean_curve(gs, key); a.plot(np.arange(1, len(c) + 1), c, color=colors[t], lw=lw, label=NAMES[t])
ax[0].set_yscale("log"); ax[0].axhline(TAU, color="gray", ls="--", lw=1)
ax[0].set_title("Training loss (eval mode, no dropout)"); ax[1].set_title("Validation loss")
for a in ax:
    a.set_xlabel("epoch"); a.xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_ylabel("loss (mean of splits; log scale)"); ax[1].set_ylabel("loss (mean of splits)")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "/memorization.png", dpi=220); print("\nsaved figure and CSVs in", OUT)
