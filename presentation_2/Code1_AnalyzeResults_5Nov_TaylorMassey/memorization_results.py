#!/usr/bin/env python3
"""
What each step does
  run      For each of the five splits it submits a job that trains the model with the base code's own
           main.py on the split files that ship with the base code (nothing is regenerated, so the P2 split
           flaw is reproduced as given). Logs go to <base code>/train_logs/split0N.out. After the five jobs
           finish, one more job rescoring them (below) runs automatically.
  rescore  Reloads every saved epoch of the five runs and recomputes each validation case's loss, -log p(true
           class). P2 weights benign cases by 2.5285, so the P2 setting applies that weight. Sanity check: the
           mean of the recomputed losses equals the validation loss logged during training. Writes
           <base code>/per_case_results/per_case_val_loss_all_splits.csv. (Run by `run`; can be run alone.)
  analyze  Reads the logs, the per-case CSVs, the test predictions and the split files, and writes ./results/:
             slide 1  best/final validation loss per split + loss curves
             slide 2  top-2 share of validation loss, accuracy, worst case + per-case figure
             slide 3  summed loss of misclassified vs correctly classified cases + stacked-bar figure
             slide 4  cases wrong in P1 validation / P1 test / P2 validation
             slide 5  case-count audit of the P1 and P2 split files, P2 validation accuracy
           plus analysis_output.txt (everything printed).
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _folder(names, fallback):
    """The base-code folder next to this script, else a fallback path."""
    for name in names:
        if os.path.isdir(os.path.join(HERE, name)):
            return os.path.join(HERE, name)
    return fallback


P1_DIR = _folder(("Presentation1_Base_Code", "Presentation_1_Base_Code"), "~/presentation_1/base_code")
P2_DIR = _folder(("Presentation2_Base_Code", "Presentation_2_Base_Code"), "~/presentation_2/base_code")
VERSIONS = {"p1": {"folder": P1_DIR, "benign_weight": 1.0},     # P1 training was effectively unweighted
            "p2": {"folder": P2_DIR, "benign_weight": 2.5285}}  # P2 weights benign cases (CrossEntropyLoss weight)
SPLITS, LOGS, OUT = "no_train_val_leakage_splits", "train_logs", "per_case_results"
ENV_SETUP = ('module purge; module load python-anaconda3/2019.10; '
             'eval "$(conda shell.bash hook)"; conda activate /projects/e32998/env')
ACCOUNT = "e32998"
PSEUDO = {107, 108, 109, 110, 111, 118, 119}      # benign pseudo-cases created by splitting four benign patients
PS_TOTAL, PS_LOW_GRADE, HG_NO_PATCHES = 105, 13, 6   # problem statement: 105 cases, 13 low-grade; walkthrough: 6 high-grade without patches
FOCUS = [56, 26, 46, 45, 94, 77, 17, 15, 92, 47]
NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
EPOCH_HEADER = re.compile(r"Epoch\s+(\d+)/\d+")
METRICS = re.compile(r"Train Loss:\s+" + NUM + r".*?Val Loss:\s+" + NUM + r"(?:.*?Val Acc:\s+" + NUM + r")?")
RUN_PATTERN = re.compile(r"(/[\w./-]*/run_\d{8}_\d{6})")


def p(path):
    return os.path.expanduser(path)


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)


def folder_of(version, override=None):
    return p(override or VERSIONS[version]["folder"])


def discover_runs(log_dir):
    """Run folder of each split = first run_... path printed in train_logs/split0N.out."""
    runs = {}
    for split in range(1, 6):
        log = os.path.join(p(log_dir), f"split0{split}.out")
        if os.path.isfile(log):
            with open(log, errors="replace") as f:
                for line in f:
                    m = RUN_PATTERN.search(line)
                    if m:
                        runs[split] = m.group(1)
                        break
    return runs


# =============================================================================== run
def sbatch(name, inner, out, err, partition, time, mem, ntasks, dependency=None):
    cmd = ["sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={partition}", f"--time={time}",
           "--nodes=1", f"--ntasks={ntasks}", f"--mem={mem}", f"--job-name={name}",
           f"--output={out}", f"--error={err}"]
    if dependency:
        cmd.append("--dependency=afterany:" + ":".join(dependency))
    cmd += ["--wrap", "bash -lc " + shlex.quote(inner)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"sbatch failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout.strip().split(";")[0]


def run(args):
    folder = folder_of(args.version, args.folder)
    splits_dir = p(args.splits_dir) if args.splits_dir else os.path.join(folder, SPLITS)
    chosen = args.only or [1, 2, 3, 4, 5]
    for s in chosen:
        f = os.path.join(splits_dir, f"data_splits_new_0{s}.npz")
        if not os.path.isfile(f):
            sys.exit(f"Split file not found: {f}\nThis script uses the split files shipped with the base code.")
    log_dir = os.path.join(folder, LOGS)
    runs_dir = p(args.runs_dir or f"~/memorization_runs/{args.version}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    script = os.path.abspath(__file__)
    print(f"base code: {folder}\nlogs: {log_dir}\nrun folders (checkpoints): {runs_dir}\n")
    ids = []
    for s in chosen:
        inner = (f"{ENV_SETUP}; python {shlex.quote(script)} launch --version {args.version} --split {s} "
                 f"--folder {shlex.quote(folder)} --splits_dir {shlex.quote(splits_dir)} "
                 f"--runs_dir {shlex.quote(runs_dir)}")
        jid = sbatch(f"{args.version}_train{s}", inner, f"{log_dir}/split0{s}.out", f"{log_dir}/split0{s}.err",
                     "long", "40:00:00", "128G", 16)
        ids.append(jid)
        print(f"submitted training job for split {s}: {jid}")
    only = (" --only " + " ".join(map(str, chosen))) if args.only else ""
    inner = (f"{ENV_SETUP}; python {shlex.quote(script)} rescore --version {args.version} "
             f"--folder {shlex.quote(folder)} --splits_dir {shlex.quote(splits_dir)}{only}")
    rid = sbatch(f"{args.version}_rescore", inner, f"{log_dir}/rescore.out", f"{log_dir}/rescore.err",
                 "short", "03:00:00", "32G", 4, dependency=ids)
    print(f"submitted rescoring job (starts after the training jobs end): {rid}")
    print("\nCheck progress with: squeue -u $USER   (logs are in the train_logs folder)")
    print("When every job has finished, run the same command for the other version, then: "
          "python memorization_results.py analyze")


def launch(args):
    """Runs inside a Slurm job: trains one split with the base code's own main.py."""
    import runpy
    import time

    folder = p(args.folder)
    os.chdir(folder)
    sys.path.insert(0, folder)
    time.sleep(20 * args.split)                    # stagger starts: run folders are named by the second
    import config

    if args.runs_dir and "runs_dir" in config.DATA_PATHS:
        os.makedirs(args.runs_dir, exist_ok=True)
        config.DATA_PATHS["runs_dir"] = args.runs_dir
    split_file = os.path.join(p(args.splits_dir), f"data_splits_new_0{args.split}.npz")
    sys.argv = ["main.py", "--per_slice_cap", "500", "--max_slices_per_stain", "5", "--load_splits", split_file]
    print(f"[launch] version={args.version} split={args.split} runs_dir={config.DATA_PATHS.get('runs_dir')}", flush=True)
    runpy.run_path("main.py", run_name="__main__")


# =============================================================================== rescore (heavy)
def find_epoch_files(ckpt_dir):
    files = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*_epoch*.pth")):
        m = re.search(r"_epoch(\d+)\.pth$", path)
        if m:
            files[int(m.group(1))] = path
    if not files:
        raise FileNotFoundError(f"No epoch checkpoints found in {ckpt_dir}")
    return dict(sorted(files.items()))


def rescore(args):
    import contextlib
    import io
    from types import SimpleNamespace

    folder = folder_of(args.version, args.folder)
    os.chdir(folder)                               # the base code's config, models, dataset, main are imported from here
    sys.path.insert(0, folder)
    import torch
    from torch.utils.data import DataLoader
    from config import DATA_PATHS, MODEL_CONFIG
    from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
    from main import EMB_DIR, prepare_data
    from models import create_model

    weight = VERSIONS[args.version]["benign_weight"] if args.benign_weight is None else args.benign_weight
    splits_dir = p(args.splits_dir) if args.splits_dir else os.path.join(folder, SPLITS)
    runs = discover_runs(args.log_dir or os.path.join(folder, LOGS))
    for item in args.run:
        key, val = item.split("=", 1)
        runs[int(key)] = p(val)
    chosen = args.only if args.only else sorted(runs)
    out_dir = args.out_dir or os.path.join(folder, OUT + ("_only" if args.only else ""))
    os.makedirs(out_dir, exist_ok=True)
    print(f"benign weight = {weight}; output folder = {out_dir}")

    frames = []
    for split in chosen:
        print(f"\n##### Split {split}: {runs[split]} #####", flush=True)
        ckpts = find_epoch_files(os.path.join(p(runs[split]), "checkpoints"))
        logged = torch.load(ckpts[max(ckpts)], map_location="cpu", weights_only=False)["val_losses"]
        prep = SimpleNamespace(labels_csv=DATA_PATHS["labels_csv"], patches_dir=DATA_PATHS["patches_dir"],
                               load_splits=os.path.join(splits_dir, f"data_splits_new_0{split}.npz"), seed=42)
        with contextlib.redirect_stdout(io.StringIO()):          # silence the long data summary
            _, (val_cases, val_labels), _ = prepare_data(prep)
        ds = StainBagCasePooledFeatureDataset(val_cases, val_labels, embeddings_dir=EMB_DIR, per_slice_cap=500,
                                              max_slices_per_stain=5, shuffle_patches=False)
        cases = [b[0] for b in DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                                          collate_fn=case_collate_fn)]
        n_patches = {c["case_id"]: sum(len(t) for sl in c["stain_slices"].values() for t in sl) for c in cases}
        model = create_model(num_classes=MODEL_CONFIG["num_classes"], embed_dim=MODEL_CONFIG["embed_dim"]).to("cpu")

        worst_gap = 0.0
        for ep, path in ckpts.items():
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
            model.eval()
            rows = []
            with torch.inference_mode():
                for c in cases:
                    logits = model(c["stain_slices"])
                    label = int(c["label"].reshape(-1)[0])
                    loss = -float(torch.log_softmax(logits, dim=-1)[label])
                    rows.append({"case_id": c["case_id"], "true_label": label,
                                 "loss": loss * (weight if label == 0 else 1.0),
                                 "prob_high_grade": float(torch.softmax(logits, dim=-1)[1]),
                                 "split": split, "epoch": ep, "n_patches": n_patches[c["case_id"]]})
            worst_gap = max(worst_gap, abs(np.mean([r["loss"] for r in rows]) - logged[ep - 1]))
            frames += rows
        print(f"  {len(cases)} val cases, epochs 1..{max(ckpts)}; sanity gap |recomputed - logged| = "
              f"{worst_gap:.5f} ({'OK' if worst_gap <= 0.01 else 'MISMATCH: check weight / splits / code folder'})",
              flush=True)
        pd.DataFrame(frames).to_csv(os.path.join(out_dir, "per_case_val_loss_all_splits.csv"), index=False)
    print(f"\nSaved {out_dir}/per_case_val_loss_all_splits.csv")


# =============================================================================== analyze (fast)
class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def load_percase(path):
    d = pd.read_csv(p(path))
    d["wrong"] = (d["prob_high_grade"] >= 0.5).astype(int) != d["true_label"].astype(int)
    return d


def best_final_epochs(d):
    """best = epoch with the lowest mean per-case loss (equals the logged validation loss); final = last epoch."""
    m = d.groupby(["split", "epoch"])["loss"].mean().reset_index()
    best = m.loc[m.groupby("split")["loss"].idxmin()].set_index("split")["epoch"].astype(int).to_dict()
    final = m.groupby("split")["epoch"].max().astype(int).to_dict()
    return best, final


def parse_log(path):
    rows, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = EPOCH_HEADER.search(line)
            if m:
                cur = int(m.group(1))
                continue
            m = METRICS.search(line)
            if m and cur is not None:
                rows.append({"epoch": cur, "train_loss": float(m.group(1)), "val_loss": float(m.group(2))})
                cur = None
    return pd.DataFrame(rows)


def slide1(args, out):
    print("\n=== Validation loss per split (from the training logs) ===")
    rows, curves = [], {}
    for path in sorted(glob.glob(os.path.join(p(args.p1_logs), "split0*.out"))):
        df = parse_log(path)
        if df.empty:
            print(f"[WARN] no epochs parsed from {path}")
            continue
        n = int(re.search(r"split0(\d)", path).group(1))
        b, f = df.loc[df["val_loss"].idxmin()], df.iloc[-1]
        rows.append({"split": n, "best_epoch": int(b["epoch"]), "best_val_loss": round(b["val_loss"], 3),
                     "final_val_loss": round(f["val_loss"], 3), "final_over_best": round(f["val_loss"] / b["val_loss"], 3),
                     "final_train_loss": round(f["train_loss"], 4)})
        curves[n] = df
    t = pd.DataFrame(rows).sort_values("split")
    print(t.to_string(index=False))
    t.to_csv(os.path.join(out, "slide1_table.csv"), index=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (n, df) in enumerate(sorted(curves.items())):
        c = plt.cm.tab10.colors[i]
        ax.plot(df["epoch"], df["train_loss"], "--", color=c, alpha=0.7, label=f"split {n} train")
        ax.plot(df["epoch"], df["val_loss"], color=c, lw=2, label=f"split {n} validation")
        j = df["val_loss"].idxmin()
        ax.scatter(df.loc[j, "epoch"], df.loc[j, "val_loss"], color=c, s=50, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs. validation loss per split (dot = best validation epoch)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slide1_curves.png"), dpi=200)
    plt.close(fig)
    return {int(r["split"]): int(r["best_epoch"]) for r in rows}


def slide2(d, best, final, log_best, out):
    print("\n=== How concentrated is the validation loss? (Presentation 1) ===")
    if log_best:
        check(all(best.get(k) == v for k, v in log_best.items()),
              "best epoch from the per-case CSV equals the best epoch in the training logs")
    rows, tops = [], {}
    for s in sorted(best):
        g = d[d["split"] == s]
        fin = g[g["epoch"] == final[s]].sort_values("loss", ascending=False)
        bst = g[g["epoch"] == best[s]]
        tops[s] = [str(c) for c in fin["case_id"].head(2)]
        rows.append({"split": s, "top2_share_of_final_val_loss": round(fin["loss"].head(2).sum() / fin["loss"].sum(), 3),
                     "accuracy_at_best_epoch": f"{int((~bst['wrong']).sum())}/{len(bst)}",
                     "worst_case_final": int(fin.iloc[0]["case_id"]), "worst_case_loss_final": round(fin.iloc[0]["loss"], 2),
                     "worst_case_p_high_grade": round(fin.iloc[0]["prob_high_grade"], 3),
                     "mean_loss_of_other_cases_final": round(fin.iloc[1:]["loss"].mean(), 4)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(out, "slide2_table.csv"), index=False)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, s in zip(axes.flat, sorted(best)):
        g = d[d["split"] == s]
        for cid, gc in g.groupby("case_id"):
            hot = str(cid) in tops[s]
            ax.plot(gc["epoch"], gc["loss"], color="tab:red" if hot else "lightgray", lw=2 if hot else 0.8,
                    zorder=3 if hot else 1, label=f"case {cid}" if hot else None)
        ax.plot(g.groupby("epoch")["loss"].mean(), "k--", label="mean of all val cases", zorder=4)
        ax.plot(g[~g["case_id"].astype(str).isin(tops[s])].groupby("epoch")["loss"].mean(), color="tab:blue", lw=2,
                label="mean without the 2 worst", zorder=4)
        ax.axvline(best[s], color="green", ls=":")
        ax.set_title(f"Split {s} (green line = best epoch)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(best):]:
        ax.axis("off")
    fig.suptitle("Per-case validation loss (gray = each case; red = 2 worst at last epoch)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slide2_per_case_loss.png"), dpi=200)
    plt.close(fig)


def slide3(d, best, final, out):
    print("\n=== Misclassified vs correctly classified cases (best epoch -> final epoch) ===")
    rows = []
    for s in sorted(best):
        g, rec = d[d["split"] == s], {"split": s}
        for tag, ep in (("best", best[s]), ("final", final[s])):
            sub = g[g["epoch"] == ep]
            rec[f"n_wrong_{tag}"] = int(sub["wrong"].sum())
            rec[f"loss_wrong_{tag}"] = round(sub.loc[sub["wrong"], "loss"].sum(), 3)
            rec[f"loss_correct_{tag}"] = round(sub.loc[~sub["wrong"], "loss"].sum(), 3)
            rec[f"median_case_loss_{tag}"] = round(sub["loss"].median(), 5)
        rec["accuracy_best"] = f"{int((~g[g.epoch == best[s]]['wrong']).sum())}/{(g.epoch == best[s]).sum()}"
        rec["accuracy_final"] = f"{int((~g[g.epoch == final[s]]['wrong']).sum())}/{(g.epoch == final[s]).sum()}"
        rows.append(rec)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(out, "slide3_decomposition.csv"), index=False)
    red, blue, w, x = "#c0392b", "#4a78b5", 0.36, np.arange(len(t))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, r in t.iterrows():
        for off, tag in ((-w / 2 - 0.02, "best"), (w / 2 + 0.02, "final")):
            wr, co = r[f"loss_wrong_{tag}"], r[f"loss_correct_{tag}"]
            ax.bar(x[i] + off, co, w, color=blue, label="Correctly classified cases" if (i == 0 and tag == "best") else None)
            ax.bar(x[i] + off, wr, w, bottom=co, color=red, label="Misclassified cases" if (i == 0 and tag == "best") else None)
            ax.text(x[i] + off, co + wr + 0.15, f"{co + wr:.1f}", ha="center", fontsize=9)
            ax.text(x[i] + off, -0.55, f"{tag}\n{int(r[f'n_wrong_{tag}'])} wrong", ha="center", va="top", fontsize=8.5, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Split {int(s)}" for s in t["split"]], fontweight="bold")
    ax.tick_params(axis="x", pad=34, length=0)
    ax.set_ylabel("Summed validation loss (all cases)")
    ax.set_title("Where the validation loss comes from: best epoch vs. final epoch", fontweight="bold")
    [ax.spines[s].set_visible(False) for s in ("top", "right")]
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slide3_loss_by_group.png"), dpi=200)
    plt.close(fig)


def slide4(args, d1, out):
    print("\n=== Cases wrong across models (last-epoch models) ===")
    parts = []

    def val_part(d, name):
        fin = d[d["epoch"] == d.groupby("split")["epoch"].transform("max")]
        return pd.DataFrame({"case_id": fin["case_id"].astype(int), "true_label": fin["true_label"].astype(int),
                             "source": name, "wrong": fin["wrong"].astype(int)})
    parts.append(val_part(d1, "P1 val"))
    runs = args.p1_runs or list(discover_runs(args.p1_logs).values())
    for run_dir in runs:
        f = os.path.join(p(run_dir), "predictions.csv")
        if os.path.isfile(f):
            t = pd.read_csv(f)
            parts.append(pd.DataFrame({"case_id": t["case_id"].astype(int), "true_label": t["true_label"].astype(int),
                                       "source": "P1 test",
                                       "wrong": (t["predicted_label"].astype(int) != t["true_label"].astype(int)).astype(int)}))
        else:
            print(f"[WARN] missing {f}")
    if os.path.isfile(p(args.p2_percase)):
        parts.append(val_part(load_percase(args.p2_percase), "P2 val"))
    long = pd.concat(parts, ignore_index=True)
    tab = long.pivot_table(index=["case_id", "true_label"], columns="source", values="wrong", aggfunc="first")
    cols = [c for c in ("P1 val", "P1 test", "P2 val") if c in tab.columns]
    tab["n_wrong"] = tab[cols].sum(axis=1)
    tab["n_assessed"] = tab[cols].notna().sum(axis=1)
    tab = tab.reset_index()
    tab["class"] = tab["true_label"].map({0: "benign", 1: "high-grade"})
    tab["patches"] = tab["case_id"].map(d1.groupby("case_id")["n_patches"].first())
    tab.to_csv(os.path.join(out, "slide4_case_difficulty.csv"), index=False)
    show = tab.copy()
    for c in cols:
        show[c] = show[c].map({1.0: "WRONG", 0.0: "ok"}).fillna("-")
    keep = ["case_id", "class", "patches"] + cols + ["n_wrong", "n_assessed"]
    print("Cases wrong in at least 2 assessments:")
    print(show[show["n_wrong"] >= 2].sort_values(["n_wrong", "case_id"], ascending=[False, True])[keep].to_string(index=False))
    print("\nFocus cases used in the slides:")
    print(show[show["case_id"].isin(FOCUS)].sort_values("case_id")[keep].to_string(index=False))
    for c in cols:
        sub = tab[tab[c].notna()]
        print(f"{c}: {int(sub[c].sum())} wrong of {len(sub)} cases assessed")


def load_splits_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(p(d), "data_splits_new_0*.npz"))):
        n = int(os.path.basename(f).split("_")[-1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        out[n] = {k.replace("_cases", ""): {int(x) for x in z[k]} for k in ("train_cases", "val_cases", "test_cases")}
    if not out:
        raise SystemExit(f"No data_splits_new_0*.npz found in {d}")
    return out


def slide5(args, d2, out):
    print("\n=== Case-count audit of the split files ===")
    uni = {}
    for name, folder in (("P1", args.p1_splits), ("P2", args.p2_splits)):
        sp = load_splits_dir(folder)
        uni[name] = set().union(*[s[k] for s in sp.values() for k in s])
        for part in ("val", "test"):
            cnt = Counter(c for s in sp.values() for c in s[part])
            check(all(cnt[c] == 1 for c in uni[name]), f"{name}: every case is in {part} exactly once")
        check(all(not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"]) for s in sp.values()),
              f"{name}: train / val / test are disjoint in all five splits")
    lab = pd.read_csv(p(args.labels_extra))
    lab.columns = [c.lower().strip() for c in lab.columns]
    cls = dict(zip(pd.to_numeric(lab["case"], errors="coerce"), pd.to_numeric(lab["class"], errors="coerce")))
    rows = []
    for name in ("P1", "P2"):
        b = sum(cls.get(c) == 1.0 for c in uni[name])
        h = sum(cls.get(c) in (3.0, 4.0) for c in uni[name])
        rows.append({"version": name, "cases": len(uni[name]), "benign": b, "high_grade": h})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(out, "slide5_case_counts.csv"), index=False)
    check(uni["P2"] <= uni["P1"] and uni["P1"] - uni["P2"] == PSEUDO,
          f"cases in P1 but not P2 are exactly the 7 pseudo-cases {sorted(uni['P1'] - uni['P2'])}")
    exp = PS_TOTAL - PS_LOW_GRADE - HG_NO_PATCHES
    check(len(uni["P2"]) == exp and len(uni["P1"]) == exp + len(PSEUDO),
          f"arithmetic: {PS_TOTAL} - {PS_LOW_GRADE} low-grade - {HG_NO_PATCHES} high-grade without patches = {exp} (P2); "
          f"+ {len(PSEUDO)} pseudo-cases = {exp + len(PSEUDO)} (P1)")
    if d2 is not None:
        fin = d2[d2["epoch"] == d2.groupby("split")["epoch"].transform("max")]
        acc = fin.groupby("split")["wrong"].agg(lambda w: f"{int((~w).sum())}/{len(w)}")
        print("P2 validation accuracy at the last epoch, splits 1-5:", ", ".join(acc.tolist()))


def analyze(args):
    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "analysis_output.txt"))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    log_best = slide1(args, args.out_dir)
    d1 = load_percase(args.p1_percase)
    best, final = best_final_epochs(d1)
    slide2(d1, best, final, log_best, args.out_dir)
    slide3(d1, best, final, args.out_dir)
    slide4(args, d1, args.out_dir)
    d2 = load_percase(args.p2_percase) if os.path.isfile(p(args.p2_percase)) else None
    slide5(args, d2, args.out_dir)
    print(f"\nSaved tables and figures in {args.out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="submit training jobs for the five splits, then a rescoring job")
    r.add_argument("--version", choices=sorted(VERSIONS), required=True)
    r.add_argument("--only", type=int, nargs="*", default=None, help="only these splits (quick test)")
    r.add_argument("--folder", default=None)
    r.add_argument("--splits_dir", default=None)
    r.add_argument("--runs_dir", default=None, help="where checkpoints go (default ~/memorization_runs/<version>)")

    l = sub.add_parser("launch", help="(used by the Slurm jobs) train one split")
    l.add_argument("--version", choices=sorted(VERSIONS), required=True)
    l.add_argument("--split", type=int, required=True)
    l.add_argument("--folder", required=True)
    l.add_argument("--splits_dir", required=True)
    l.add_argument("--runs_dir", default=None)

    s = sub.add_parser("rescore", help="recompute per-case losses from the saved checkpoints")
    s.add_argument("--version", choices=sorted(VERSIONS), required=True)
    s.add_argument("--only", type=int, nargs="*", default=None)
    s.add_argument("--folder", default=None)
    s.add_argument("--log_dir", default=None)
    s.add_argument("--splits_dir", default=None)
    s.add_argument("--run", nargs="*", default=[], help="override a run folder, e.g. --run 3=/path/run_X")
    s.add_argument("--out_dir", default=None)
    s.add_argument("--benign_weight", type=float, default=None)

    a = sub.add_parser("analyze", help="build every slide table and figure")
    a.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    a.add_argument("--p1_logs", default=os.path.join(P1_DIR, LOGS))
    a.add_argument("--p1_percase", default=os.path.join(P1_DIR, OUT, "per_case_val_loss_all_splits.csv"))
    a.add_argument("--p2_percase", default=os.path.join(P2_DIR, OUT, "per_case_val_loss_all_splits.csv"))
    a.add_argument("--p1_runs", nargs="*", default=None, help="P1 run folders; default: found from the P1 logs")
    a.add_argument("--p1_splits", default=os.path.join(P1_DIR, SPLITS))
    a.add_argument("--p2_splits", default=os.path.join(P2_DIR, SPLITS))
    a.add_argument("--labels_extra", default=os.path.join(P1_DIR, "extra_benign_case_grade_match.csv"))

    args = ap.parse_args()
    {"run": run, "launch": launch, "rescore": rescore, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
