# Yifan Xia: Presentation 1 Coder Submission

This folder contains the source code, five case-grouped splits, validation records, and test results for Yifan Xia's completed Stage 3 and Stage 4 experiments. It is a coder contribution to Team 3's Presentation 1. The team's final slides and documentation belong in the parent `presentation_1/` folder and are maintained separately.

## Results at a glance

All five runs completed on Quest on October 22, 2026. Seed: **3902026**. Labels: **0 = benign, 1 = high-grade**.

| Split | Train / Val / Test cases | Accuracy | Benign recall | High-grade recall | Epochs | Main runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| 01 | 57 / 18 / 18 | 88.89% | 85.71% | 90.91% | 10 | 501.02 |
| 02 | 56 / 18 / 19 | 89.47% | 71.43% | 100.00% | 15 | 723.12 |
| 03 | 55 / 19 / 19 | 89.47% | 85.71% | 91.67% | 22 | 1121.91 |
| 04 | 55 / 19 / 19 | 78.95% | 85.71% | 75.00% | 14 | 697.26 |
| 05 | 56 / 19 / 18 | 72.22% | 57.14% | 81.82% | 17 | 786.42 |

Mean split accuracy is **83.80%**, with a sample SD of **7.87 percentage points**. Across the 93 unique held-out cases, pooled accuracy is **78/93 = 83.87%**. These averages use different weighting. Main runtime includes data preparation, training, validation, and testing, and excludes queue time and reporting. Parallel job runtimes overlap.

**Checkpoint policy:** testing evaluates the final in-memory epoch in each split. The provided `main.py` does not automatically reload `best.pth`. All five results preserve that behavior.

## Folder guide

- `code/`: exact executed Python sources and original launch scripts, copied from the personal Quest workspace. Inherited Stage 1/2 and Optuna utilities are included for context; they were not rerun for these experiments.
- `splits/`: the five exact `.npz` input files used for training, validation, and testing.
- `analysis/`: case assignments, class counts, original validation records, runtime/provenance manifests, and reporting helpers.
- `analysis/stage3_originals/` and `analysis/stage4_originals/`: baseline file backups for reviewing changes.
- `analysis/stage3_changes.diff`: a readable comparison of the original Stage 3 files and the executed versions.
- `results/five_split_summary/`: all-split metrics, all **93 case probabilities**, loss curves, and Slurm completion evidence.
- `results/split01/` through `results/split05/`: each split's predictions, confusion matrix, result text, epoch history, loss curves, metrics, and execution manifest.
- `figures/`: English presentation figures, including the pipeline flowchart and five-split performance and confusion-matrix summaries.
- `scripts/`: offline verification and launch instructions for a separate reproduction workspace.
- `MANIFEST.sha256`: checksums of the submitted files.

The original Quest workspace is `/home/siw1333/Yifan_Xia_coder_p1`. Its full checkpoints and raw logs remain there. Historical absolute paths in JSON records identify the original experiment and are intentionally preserved. They are not paths inside this submission folder. `code/README.md` is inherited baseline documentation; this top-level README describes the experiment actually executed.

## Source attribution and changes

The course baseline came from:

`/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation1_base_code`

The model and training implementation are the provided baseline, not a new architecture written for this submission. `main.py`, `models.py`, `trainer.py`, and the other Python sources retain the hashes recorded in the completed run manifests.

Changes in the personal copy:

1. `code/make_splits.py`: enabled `StratifiedGroupKFold(shuffle=True, random_state=seed)`. The original constructor omitted these arguments and disabled shuffling, with a comment about class-ratio deviations. This is an explicit methodological change, not merely a path change. Its purpose was to make fold assignment depend on the chosen seed. The achieved high-grade proportions were **61.11% to 63.16%**, compared with **58/93 = 62.37%** overall. These checks describe this seed and dataset, not a general guarantee about shuffling or an assertion of instructor approval.
2. `code/config.py`: set seed `3902026` and a personal output directory.
3. Slurm scripts: personal paths and resource requests. Added a runner to record provenance and isolate each split under `runs/splitNN/`; the model and training algorithm were preserved. The split 01 output directory was subsequently relocated, with file contents unchanged and the relocation recorded in its manifest.
4. Added split validation, result verification, aggregation, and English presentation figures. The shared course baseline and input datasets were not changed.

Stage 3 checks cover complete case coverage, disjoint subsets, related-case grouping, class composition, same-seed reproducibility, and differences from Taylor's five reference assignments. Only Taylor's reference was compared, not every team. The validator's broad acceptance threshold (absolute fraction difference <= 0.15) is a helper setting, not a course requirement. The actual maximum deviation from the overall high-grade fraction was approximately **1.25 percentage points**. Model accuracy was not used to select the seed.

## Inputs and environment

These are shared Quest inputs, not included as copies in Git:

| Input | Quest location |
|---|---|
| Labels used | `/projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/extra_benign_case_grade_match.csv` |
| Patch inventory | `/projects/e32998/patches_benign_split` |
| Precomputed pooled features | `/projects/e32998/patches_varsize_pooled4096` |
| Course Python | `/projects/e32998/env/bin/python` |
| Original pretrained weights (Stage 2 only) | `/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation1_base_code/KimiaNet.pth` |

Recorded environment: Python 3.12.12, PyTorch 2.9.0, NumPy 2.3.4, and scikit-learn 1.8.0. See `code/requirements.txt` for the inherited dependency list. That list specifies broad minimum versions, not an exact environment lock. Fixed seeds support reproduction within the recorded environment; exact results across different software/hardware are not guaranteed.

Stage 1 patches and Stage 2 frozen KimiaNet/DenseNet-121 pooled features were provided upstream. Stage 4 uses 4096-dimensional patch features and a hierarchical attention MIL classifier with a 512-dimensional projection. The five runs each start a new model. Settings include batch size 1, two data-loader workers, up to 500 patches per slice, up to five slices per stain, AdamW learning rate 0.000208, weight decay 0.000305, dropout 0.2599, class weights [2.5285, 1.0], and normalized patch-entropy regularization in max mode with weight 0.001. The maximum is 30 epochs, with early stopping (minimum 10 epochs, patience 8, minimum improvement 0.001). Each training job requests 8 CPUs and 32 GB RAM on the `short` partition, with CUDA disabled.

## Inspect the submitted results without retraining

From this folder, using Python with NumPy installed:

```bash
/projects/e32998/env/bin/python scripts/verify_submission.py
```

This verifies submitted file hashes, saved split assignments, family separation, all 93 prediction probabilities, per-split confusion matrices and metrics, and source-code hashes against the original execution manifests. It does not require raw images, model weights, a GPU, Slurm accounting access, or training. This export check is separate from the original full-workspace validation.

The complete test probability table is `results/five_split_summary/out_of_fold_predictions.csv`. Columns are `split`, `case_id`, `true_label`, `predicted_label`, `correct`, `prob_class0`, and `prob_class1`. Confusion matrices have true classes in rows and predicted classes in columns, ordered benign then high-grade.

## Reproduce Stages 3 and 4 in a fresh Quest workspace

The following are instructions for a future rerun. Creating this submission did **not** launch new training. The submitted files and completed workspace should remain unchanged.

From this submission folder on Quest:

```bash
export RUN_ROOT="$HOME/Yifan_Xia_coder_p1_reproduction"
/projects/e32998/env/bin/python scripts/prepare_workspace.py --workspace "$RUN_ROOT"
```

The preparation helper refuses an existing destination, copies the exact source and analysis helpers, and changes only the new copy's `config.DATA_PATHS['runs_dir']`. Source hashes are regenerated by Stage 3 validation in that new workspace. Shared input locations must remain accessible. The original `.sbatch` files in `code/` record historical personal paths; use the submission launchers below for this new workspace.

Stage 3 uses the original Taylor reference for the comparison check. Set `REFERENCE_SPLITS` to a directory containing her five `data_splits_new_*.npz` files if that reference has moved:

```bash
export REFERENCE_SPLITS=/projects/e32998/STAT390_Fall2026/Team3_DivideNConquer/presentation_1/taylor_code/no_train_val_leakage_splits
sbatch --export=ALL,RUN_ROOT,REFERENCE_SPLITS \
  --output="$RUN_ROOT/logs/stage3_%j.out" \
  --error="$RUN_ROOT/logs/stage3_%j.err" scripts/reproduce_stage3.sbatch
```

Wait for Stage 3 to complete successfully and confirm that the new `analysis/split_validation.json` reports `all_checks_passed: true`. Then submit five independent Stage 4 runs:

```bash
sbatch --export=ALL,RUN_ROOT \
  --output="$RUN_ROOT/logs/stage4_%A_%a.out" \
  --error="$RUN_ROOT/logs/stage4_%A_%a.err" scripts/reproduce_stage4.sbatch
```

After all five jobs complete, aggregate in the new workspace:

```bash
/projects/e32998/env/bin/python "$RUN_ROOT/analysis/training_tools/aggregate_training.py" \
  --workspace "$RUN_ROOT" --output "$RUN_ROOT/analysis/five_split_summary"
```

The original aggregate helper validates checkpoints in the full Quest workspace and queries Slurm accounting. It is intended for a completed rerun or the original full workspace. It is not an offline reader for this export, which intentionally excludes checkpoints and rearranges small results under `results/`.

## Files intentionally kept on Quest

Raw images, shared feature tensors, `KimiaNet.pth`, all training checkpoints, and full raw logs are excluded from this Git submission. No original training output was deleted during packaging. Successful job IDs were 4841885, 4844082_2, 4844082_3, 4844082_4, and 4844082_5; Stage 3 was job 4814175. Their saved accounting evidence is included with the results.
