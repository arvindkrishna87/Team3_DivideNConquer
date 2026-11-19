import glob, json, os, re
import numpy as np

LOGS = "/home/vnc0125/presentation_3/taylor_logs/exp99_*.out"
TAU = 0.05
ORDER = ["baseline", "stain0.3", "slice0.3", "emb64", "emb128", "emb256"]

runs = {}
for lf in sorted(glob.glob(LOGS), key=os.path.getmtime):
    txt = open(lf, errors="ignore").read()
    m = re.search(r"Task (\d+) \| tag=(\S+) \| split=(\S+)", txt)
    d = re.search(r"Created run directory: (\S+)", txt)
    if m and d:
        runs[(m.group(2), int(m.group(3)[-6:-4]))] = d.group(1)
info = {}
for (tag, sp), rd in runs.items():
    hf = rd + "/history.json"
    if os.path.isfile(os.path.dirname(rd) + "/FINISHED") and os.path.isfile(hf):
        h = json.load(open(hf)); val, cl = np.array(h["val_losses"]), np.array(h["train_clean_losses"])
        b = int(np.argmin(val)); below = np.where(cl < TAU)[0]
        info[(tag, sp)] = dict(tau=int(below[0]) + 1 if len(below) else None, best=b + 1,
                               val=float(val[b]), clean=float(cl[b]), n=len(val))

print("Epoch where clean training loss first falls below %.2f, by split" % TAU)
print("%-10s %s" % ("setting", "  ".join("split%d" % s for s in range(1, 6))))
for t in ORDER:
    print("%-10s %s" % (t, "  ".join("%6s" % (info[(t, s)]["tau"] if (t, s) in info else "-") for s in range(1, 6))))

print("\nLater than baseline on how many of the 5 splits (strictly later / same / earlier)?")
for t in ORDER[1:]:
    later = same = earlier = 0
    for s in range(1, 6):
        if (t, s) in info and ("baseline", s) in info and info[(t, s)]["tau"] and info[("baseline", s)]["tau"]:
            d = info[(t, s)]["tau"] - info[("baseline", s)]["tau"]
            later += d > 0; same += d == 0; earlier += d < 0
    print("%-10s later %d / same %d / earlier %d" % (t, later, same, earlier))

print("\nAt each run's best-validation epoch (mean over splits)")
print("%-10s %9s %12s %8s %10s" % ("setting", "best val", "train(clean)", "gap", "best epoch"))
for t in ORDER:
    rs = [info[(t, s)] for s in range(1, 6) if (t, s) in info]
    if rs:
        v, c = np.mean([r["val"] for r in rs]), np.mean([r["clean"] for r in rs])
        print("%-10s %9.3f %12.3f %8.3f %10.1f" % (t, v, c, v - c, np.mean([r["best"] for r in rs])))
