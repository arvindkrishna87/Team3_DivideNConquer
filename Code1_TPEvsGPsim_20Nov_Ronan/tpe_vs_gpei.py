#!/usr/bin/env python3
"""
Toy model simulation

--------------------------------------------------------------------------------------------------------------------------------
CODED WITH CLAUDE OPUS 5.5
--------------------------------------------------------------------------------------------------------------------------------

Question: given the SAME objective, the SAME initial trials and the SAME extra trial budget,
  (1) do TPE's evaluations cluster around the best initial settings, and
  (2) which method more often evaluates a setting within EPS of the global minimum, when that minimum lies far from every initial setting?

Objective:
    f(x) = 0.65 - 0.30 exp(-(x1 - 3)^2 / (2 * 0.7^2)) - 0.60 exp(-(x1 - 8)^2 / (2 * w^2))
    local minimum near x1 = 3 (f ~ 0.35), global minimum at x1 = 8 (f = 0.05).
    w = width of the global basin (default 0.4; change with --basin_width).
    In the 6-D and 10-D settings, x2 ... xd are irrelevant hyperparameters (like many real ones).

Methods (all share the same initial trials within a repeat; all GP-EI variants share the same noise draws)
    TPE            Optuna TPESampler, library defaults (prior on), independent (univariate) TPE.
    TPE-weak-prior Same, but prior_weight=0.001 (default 1). Ablation of the flat prior in l(x), g(x).
                   Note: this ablation's numbers vary noticeably between Optuna versions.
    GP-EI          GP with Matern-5/2 kernel, one length scale per dimension (ARD), hyperparameters fitted by marginal likelihood (scikit-learn) after every trial by default.
                   EI(x) = (b - xi - mu) Phi(z) + sigma phi(z),  z = (b - xi - mu) / sigma,  with xi = 0.
                   Incumbent b: best observed loss (noise-free) or best posterior mean at a tried setting (noisy). Noisy settings: sigma includes the fitted observation noise.
                   Next trial = argmax EI (1-D: 2,001-point grid; d > 1: 4,000 uniform candidates +
                   1,000 Gaussian perturbations of the best 3 trials).
    GP-EI-xi       Same as GP-EI but with exploration offset xi > 0 (default 0.01, --xi).
    GP-EI-latent   Noisy settings only: GP-EI with sigma of the noise-free (latent) loss,
                   sigma_latent^2 = sigma^2 - fitted noise variance (xi = 0, same incumbent).
    GP-EI-minobs   Noisy settings only: GP-EI (noise-inclusive sigma, xi = 0) with incumbent b = lowest NOISY observation. 
                   Biased low since it picks the luckiest noise draw, so it acts like an extra exploration offset.
    Random         Uniform random search, as a baseline.

Initial trials: spreads x1 evenly over [2.0, 3.9], all near the local minimum, change withb --n_init
Success = at least one evaluated setting (initial or new) whose TRUE loss is within epsilon of f*.

GP diagnostics: for every GP-EI variant, over all steps of all repeats taken while the best trial so far is still the LOCAL minimum (|x1 - 3| <= 1), 
the script reports the median fitted prior sd and the share of those steps whose next trial lies within 0.5 of the best trial's x1 (refining the local minimum instead of exploring).

Optional GP ablations (apply to every GP-EI variant):
    --gp_refit_every K   refit the kernel hyperparameters every K trials (default 1 = every trial)
    --gp_isotropic       one shared length scale for all dimensions instead of ARD

Usage
    python tpe_vs_gpei.py --settings fixed6d --n_init 16 --outdir results/fixed6d_init16
    python tpe_vs_gpei.py --merge results
"""

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import optuna
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

LOW, HIGH = 0.0, 10.0
GRID_1D = np.linspace(LOW, HIGH, 2001)
NOISE_SD = 0.03
NOISE_LEVEL_BOUNDS = (1e-6, 10.0)          # v2 used (1e-6, 1e-1), which was too tight
ALL_METHODS = ["TPE", "TPE-weak-prior", "GP-EI", "GP-EI-xi", "GP-EI-latent", "GP-EI-minobs", "Random"]
NOISY_ONLY = {"GP-EI-latent", "GP-EI-minobs"}
SETTINGS = {  # name: (dimensions, noisy, initial x1 fixed?)
    "fixed": (1, False, True),
    "noisy": (1, True, True),
    "random": (1, False, False),
    "fixed6d": (6, False, True),
    "fixed10d": (10, False, True),
}


def methods_for(setting):
    noisy = SETTINGS[setting][1]
    return [m for m in ALL_METHODS if m not in NOISY_ONLY or noisy]


def fixed_init_x1(n):
    return [2.0, 2.6, 3.3, 3.9] if n == 4 else list(np.linspace(2.0, 3.9, n))


# ----------------------------------------------------------------------------
# Objective: depends on the first coordinate only
# ----------------------------------------------------------------------------
def make_objective(basin_width):
    def f1(x1):
        x1 = np.asarray(x1, dtype=float)
        return (0.65
                - 0.30 * np.exp(-(x1 - 3.0) ** 2 / (2 * 0.7 ** 2))
                - 0.60 * np.exp(-(x1 - 8.0) ** 2 / (2 * basin_width ** 2)))
    vals = f1(GRID_1D)
    return f1, float(vals.min()), float(GRID_1D[np.argmin(vals)])


# ----------------------------------------------------------------------------
# TPE and random search
# ----------------------------------------------------------------------------
def run_tpe(init, n_new, observe, seed, prior_weight=1.0):
    d = init.shape[1]
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=len(init),
                                         prior_weight=prior_weight, multivariate=False)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for row in init:
        study.enqueue_trial({f"x{j}": float(row[j]) for j in range(d)})

    def objective(trial):
        x = np.array([trial.suggest_float(f"x{j}", LOW, HIGH) for j in range(d)])
        return float(observe(x))

    study.optimize(objective, n_trials=len(init) + n_new)
    return np.array([[t.params[f"x{j}"] for j in range(d)] for t in study.trials])


def run_random(init, n_new, seed):
    rng = np.random.RandomState(seed)
    return np.vstack([init, rng.uniform(LOW, HIGH, (n_new, init.shape[1]))])


# ----------------------------------------------------------------------------
# GP-EI
# ----------------------------------------------------------------------------
def expected_improvement(mu, sigma, best, xi=0.0):
    """EI for minimisation (the slide's formula, with improvement threshold best - xi);
    EI = max(best - xi - mu, 0) where sigma = 0."""
    target = best - xi
    ei = np.maximum(target - mu, 0.0)
    pos = sigma > 1e-12
    z = (target - mu[pos]) / sigma[pos]
    ei[pos] = (target - mu[pos]) * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    return ei


def make_kernel(d, noisy, isotropic):
    ls = 1.0 if (d == 1 or isotropic) else np.ones(d)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=ls,
                                                       length_scale_bounds=(0.05, 100.0), nu=2.5)
    if noisy:
        kernel = kernel + WhiteKernel(1e-3, NOISE_LEVEL_BOUNDS)
    return kernel


def fit_gp(Xa, ya, noisy, rng, gp_opts, kernel_fixed=None):
    if kernel_fixed is not None:      # reuse earlier hyperparameters, only update the posterior
        return GaussianProcessRegressor(kernel=kernel_fixed, alpha=1e-8, normalize_y=True,
                                        optimizer=None).fit(Xa, ya)
    return GaussianProcessRegressor(kernel=make_kernel(Xa.shape[1], noisy, gp_opts["isotropic"]),
                                    alpha=1e-8, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=rng).fit(Xa, ya)


def gp_sigma(gp, cand, ya, noisy, latent):
    mu, sigma = gp.predict(cand, return_std=True)
    if noisy and latent:
        # predictive variance = latent variance + noise variance; normalize_y scales by std(y)^2
        noise_var = gp.kernel_.k2.noise_level * np.std(ya) ** 2
        sigma = np.sqrt(np.maximum(sigma ** 2 - noise_var, 0.0))
    return mu, sigma


def incumbent(gp, Xa, ya, noisy, kind):
    if not noisy or kind == "min_observed":
        return float(ya.min())
    return float(gp.predict(Xa).min())           # best posterior mean at a tried setting


def candidates(Xa, ya, rng):
    d = Xa.shape[1]
    if d == 1:
        return GRID_1D[:, None]
    top = Xa[np.argsort(ya)[:3]]
    local = top[rng.randint(0, len(top), 1000)] + rng.normal(0, 0.5, (1000, d))
    return np.clip(np.vstack([rng.uniform(LOW, HIGH, (4000, d)), local]), LOW, HIGH)


def run_gpei(init, n_new, observe, seed, noisy, gp_opts, xi=0.0, latent=False,
             incumbent_kind="posterior_mean"):
    X = [row.copy() for row in init]
    y = [float(observe(row)) for row in X]
    rng = np.random.RandomState(seed)
    kernel_fixed = None
    steps = []   # per step: (fitted prior sd, next trial within 0.5 of incumbent's x1?, incumbent at local min?)
    for t in range(n_new):
        Xa, ya = np.array(X), np.array(y)
        refit = kernel_fixed is None or t % gp_opts["refit_every"] == 0
        gp = fit_gp(Xa, ya, noisy, rng, gp_opts, None if refit else kernel_fixed)
        if refit:
            kernel_fixed = gp.kernel_
        cand = candidates(Xa, ya, rng)
        mu, sigma = gp_sigma(gp, cand, ya, noisy, latent)
        best = incumbent(gp, Xa, ya, noisy, incumbent_kind)
        x_next = cand[int(np.argmax(expected_improvement(mu, sigma, best, xi)))]
        k = gp.kernel_.k1 if noisy else gp.kernel_
        best_x1 = Xa[np.argmin(ya), 0]
        steps.append((float(np.sqrt(k.k1.constant_value) * np.std(ya)),
                      bool(abs(x_next[0] - best_x1) <= 0.5),
                      bool(abs(best_x1 - 3.0) <= 1.0)))
        X.append(x_next.copy())
        y.append(float(observe(x_next)))
    return np.array(X), steps


# ----------------------------------------------------------------------------
# Experiment loop
# ----------------------------------------------------------------------------
def run_setting(name, f1, args, gp_opts):
    d, noisy, fixed_x1 = SETTINGS[name]
    methods = methods_for(name)
    out = {m: [] for m in methods}
    gp_steps = {m: [] for m in methods if m.startswith("GP-EI")}
    for r in range(args.repeats):
        rng = np.random.RandomState(10_000 + r)
        init = rng.uniform(LOW, HIGH, (args.n_init, d))
        if fixed_x1:
            init[:, 0] = fixed_init_x1(args.n_init)

        def make_observe(seed):
            nrng = np.random.RandomState(seed)
            if noisy:
                return lambda x: float(f1(x[0])) + nrng.normal(0.0, NOISE_SD)
            return lambda x: float(f1(x[0]))

        def gp(name, **kw):   # every GP-EI variant sees the same noise draws (seed 30_000 + r)
            X, steps = run_gpei(init, args.n_new, make_observe(30_000 + r), seed=r, noisy=noisy,
                                gp_opts=gp_opts, **kw)
            gp_steps[name].extend(steps)
            return X

        runs = {
            "TPE": run_tpe(init, args.n_new, make_observe(20_000 + r), seed=r),
            "TPE-weak-prior": run_tpe(init, args.n_new, make_observe(25_000 + r), seed=r,
                                      prior_weight=1e-3),
            "GP-EI": gp("GP-EI"),
            "GP-EI-xi": gp("GP-EI-xi", xi=args.xi),
            "Random": run_random(init, args.n_new, seed=40_000 + r),
        }
        if noisy:
            runs["GP-EI-latent"] = gp("GP-EI-latent", latent=True)
            runs["GP-EI-minobs"] = gp("GP-EI-minobs", incumbent_kind="min_observed")
        for m in methods:
            out[m].append(runs[m][:, 0].tolist())        # keep x1, the coordinate that matters
        if (r + 1) % max(1, args.repeats // 5) == 0:
            print(f"    {name} (n_init={args.n_init}): {r + 1}/{args.repeats} repeats done", flush=True)
    # Only steps taken while the best trial so far is still the LOCAL minimum (|x1 - 3| <= 1):
    # does GP-EI keep refining there, and how much prior uncertainty has it fitted?
    diag = {}
    for m, v in gp_steps.items():
        stuck = [st for st in v if st[2]]
        diag[m] = {"steps_with_local_incumbent": len(stuck),
                   "median_prior_sd_while_local": float(np.median([st[0] for st in stuck])) if stuck else None,
                   "frac_refining_local_incumbent": float(np.mean([st[1] for st in stuck])) if stuck else None}
    return out, diag


def wilson(p, n, z=1.96):
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, c - h), min(1.0, c + h)


def summarise(x1_runs, f1, f_star, args):
    X = np.array(x1_runs)                               # (R, n_init + n_new)
    hit = f1(X) <= f_star + args.eps
    success_by_t = np.maximum.accumulate(hit, axis=1).mean(axis=0)
    new = X[:, args.n_init:]
    early = new[:, :16]
    first_hits = [int(np.argmax(h)) + 1 for h in hit if h.any()]
    return {
        "success_after_16_new": float(success_by_t[args.n_init + min(16, args.n_new) - 1]),
        "success_final": float(success_by_t[-1]),
        "success_by_trial": success_by_t.tolist(),
        "first16_frac_near_x3": float(np.mean(np.abs(early - 3.0) <= 1.0)),
        "first16_frac_near_x8": float(np.mean(np.abs(early - 8.0) <= 1.0)),
        "all_new_frac_near_x3": float(np.mean(np.abs(new - 3.0) <= 1.0)),
        "all_new_frac_near_x8": float(np.mean(np.abs(new - 8.0) <= 1.0)),
        "median_trial_of_first_success": float(np.median(first_hits)) if first_hits else None,
    }


def print_table(title, summ, repeats, diag=None):
    print(f"\n[{title}]")
    print(f"  {'method':13s} {'P(success), 16 new':>22s} {'P(success), final':>22s}"
          f" {'first 16 new: near x=3':>23s} {'near x=8':>9s}")
    for m, dct in summ.items():
        a, b = dct["success_after_16_new"], dct["success_final"]
        la, ha = wilson(a, repeats)
        lb, hb = wilson(b, repeats)
        print(f"  {m:13s} {a:6.2f} [{la:4.2f}, {ha:4.2f}]      {b:6.2f} [{lb:4.2f}, {hb:4.2f}]"
              f" {dct['first16_frac_near_x3']:23.2f} {dct['first16_frac_near_x8']:9.2f}")
    if diag:
        print("  GP-EI steps taken while the best trial was still the local minimum (all repeats):")
        print(f"    {'method':13s} {'steps':>6s} {'median prior sd':>16s} {'share refining it':>18s}")
        for m, d in diag.items():
            if d["steps_with_local_incumbent"]:
                print(f"    {m:13s} {d['steps_with_local_incumbent']:6d} {d['median_prior_sd_while_local']:16.3f}"
                      f" {d['frac_refining_local_incumbent']:18.2f}")
            else:
                print(f"    {m:13s} {0:6d} {'-':>16s} {'-':>18s}")


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
COLORS = {"TPE": "#4E2A84", "TPE-weak-prior": "#AFA9EC", "GP-EI": "#D85A30", "GP-EI-xi": "#1D9E75",
          "GP-EI-latent": "#3B8BD4", "GP-EI-minobs": "#BA7517", "Random": "#888780"}
STYLES = {"GP-EI-xi": "--", "GP-EI-latent": "--", "GP-EI-minobs": ":"}


def plot_locations(xs, f1, setting, args, path):
    n_init = args.n_init
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), gridspec_kw={"height_ratios": [1.2, 1]},
                             sharex=True)
    ax = axes[0]
    ax.axvspan(7.0, 9.0, color="#EEEDFE", zorder=0)
    ax.plot(GRID_1D, f1(GRID_1D), color="black", lw=1.5, label="objective f(x)")
    first = np.array(xs["TPE"][0][:n_init])
    ax.scatter(first, f1(first), s=50, color="black", zorder=5, label="initial trials")
    shown = [m for m in ("TPE", "GP-EI", "GP-EI-xi") if m in xs]
    for m, off in zip(shown, (-0.04, -0.09, -0.14)):
        new = np.array(xs[m][0][n_init:n_init + 16])
        ax.scatter(new, np.full_like(new, off), s=24, color=COLORS[m], alpha=0.85,
                   label=f"{m}: first 16 new trials, run 1")
    ax.set_ylabel("loss")
    ax.set_title(f"Where each method evaluates ({setting}, {n_init} initial trials)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax = axes[1]
    bins = np.linspace(LOW, HIGH, 51)
    for m in xs:
        new = np.array([run[n_init:] for run in xs[m]]).ravel()
        ax.hist(new, bins=bins, histtype="step", lw=1.8, color=COLORS[m], density=True, label=m,
                ls=STYLES.get(m, "-"))
    ax.set_xlabel("hyperparameter x (x1 in the 6-D and 10-D settings)")
    ax.set_ylabel(f"density of all {args.n_new}\nnew trials, all runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_success_grid(entries, path):
    """entries: list of dicts with keys setting, n_init, repeats, eps, summary (method -> stats)."""
    settings = [s for s in SETTINGS if any(e["setting"] == s for e in entries)]
    inits = sorted({e["n_init"] for e in entries})
    fig, axes = plt.subplots(len(inits), len(settings), figsize=(4.0 * len(settings), 3.5 * len(inits)),
                             sharey=True, squeeze=False)
    for i, n0 in enumerate(inits):
        for j, s in enumerate(settings):
            ax = axes[i][j]
            match = [e for e in entries if e["setting"] == s and e["n_init"] == n0]
            if not match:
                ax.axis("off")
                continue
            e = match[0]
            for m, dct in e["summary"].items():
                curve = np.array(dct["success_by_trial"])
                t = np.arange(1, len(curve) + 1)
                lo, hi = zip(*[wilson(p, e["repeats"]) for p in curve])
                ax.plot(t, curve, color=COLORS[m], lw=2, ls=STYLES.get(m, "-"), label=m)
                ax.fill_between(t, lo, hi, color=COLORS[m], alpha=0.10)
            ax.axvline(n0 + 0.5, color="black", ls=":", lw=1)
            ax.axvline(n0 + 16.5, color="black", ls="--", lw=0.8)
            ax.set_title(f"{s}, {n0} initial trials", fontsize=11)
            ax.set_ylim(-0.02, 1.02)
            if i == len(inits) - 1:
                ax.set_xlabel("trials evaluated (incl. initial)")
            if j == 0:
                ax.set_ylabel(f"P(within {e['eps']} of global min)")
    handles = {}
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def merge(root):
    entries = []
    for p in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        with open(p) as fh:
            d = json.load(fh)
        for s, summ in d["summary"].items():
            entries.append({"setting": s, "n_init": d["args"]["n_init"], "repeats": d["args"]["repeats"],
                            "eps": d["args"]["eps"], "xi": d["args"].get("xi"),
                            "optuna": d.get("optuna_version"), "summary": summ,
                            "gp_diagnostics": d.get("gp_diagnostics", {}).get(s)})
    if not entries:
        raise SystemExit(f"No */summary.json found under {root}")
    order = {s: i for i, s in enumerate(SETTINGS)}
    entries.sort(key=lambda e: (e["n_init"], order[e["setting"]]))
    for e in entries:
        print_table(f"{e['setting']}, {e['n_init']} initial trials, R = {e['repeats']}, "
                    f"optuna {e['optuna']}", e["summary"], e["repeats"], e["gp_diagnostics"])
    plot_success_grid(entries, os.path.join(root, "success_curves_all.png"))
    with open(os.path.join(root, "summary_all.json"), "w") as fh:
        json.dump(entries, fh, indent=2)
    print(f"\nSaved {os.path.join(root, 'success_curves_all.png')} and summary_all.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=4)
    ap.add_argument("--n_new", type=int, default=46)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--basin_width", type=float, default=0.4)
    ap.add_argument("--xi", type=float, default=0.01, help="exploration offset for GP-EI-xi")
    ap.add_argument("--gp_refit_every", type=int, default=1, help="refit GP hyperparameters every K trials")
    ap.add_argument("--gp_isotropic", action="store_true", help="one shared length scale (no ARD)")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--merge", metavar="ROOT", help="combine ROOT/*/summary.json into one table and figure")
    args = ap.parse_args()

    if args.merge:
        merge(args.merge)
        return

    gp_opts = {"refit_every": max(1, args.gp_refit_every), "isotropic": args.gp_isotropic}
    os.makedirs(args.outdir, exist_ok=True)
    f1, f_star, x_star = make_objective(args.basin_width)
    print(f"optuna {optuna.__version__}; global min f* = {f_star:.4f} at x1 = {x_star:.3f}; "
          f"eps = {args.eps}; {args.n_init} initial + {args.n_new} new trials; R = {args.repeats}; "
          f"xi = {args.xi}; GP refit every {gp_opts['refit_every']}; isotropic = {gp_opts['isotropic']}")

    all_results, summary, diags = {}, {}, {}
    for s in args.settings:
        t0 = time.time()
        all_results[s], diags[s] = run_setting(s, f1, args, gp_opts)
        summary[s] = {m: summarise(v, f1, f_star, args) for m, v in all_results[s].items()}
        print_table(f"{s}, {args.n_init} initial trials  ({time.time() - t0:.0f}s)", summary[s],
                    args.repeats, diags[s])
        plot_locations(all_results[s], f1, s, args, os.path.join(args.outdir, f"locations_{s}.png"))

    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump({"optuna_version": optuna.__version__, "args": vars(args), "f_star": f_star,
                   "x_star": x_star, "summary": summary, "gp_diagnostics": diags}, fh, indent=2)
    with open(os.path.join(args.outdir, "evaluated_x.json"), "w") as fh:
        json.dump(all_results, fh)
    print(f"\nSaved figures and JSON to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
