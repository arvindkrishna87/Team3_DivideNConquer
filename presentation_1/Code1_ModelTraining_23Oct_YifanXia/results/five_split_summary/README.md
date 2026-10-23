# Yifan Xia — Stage 4: Five-Split Results

All five baseline training, validation, and test runs completed successfully on Quest.

| Split | Train / Val / Test | Accuracy | Benign recall | High-grade recall | Epochs | Main runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| 01 | 57 / 18 / 18 | 88.89% | 85.71% | 90.91% | 10 | 501.02 |
| 02 | 56 / 18 / 19 | 89.47% | 71.43% | 100.00% | 15 | 723.12 |
| 03 | 55 / 19 / 19 | 89.47% | 85.71% | 91.67% | 22 | 1121.91 |
| 04 | 55 / 19 / 19 | 78.95% | 85.71% | 75.00% | 14 | 697.26 |
| 05 | 56 / 19 / 18 | 72.22% | 57.14% | 81.82% | 17 | 786.42 |

## Across-split summary

Mean ± sample SD across five splits (percentage points for SD):

- Accuracy: 83.80% ± 7.87 percentage points.
- Benign recall: 77.14% ± 12.78 percentage points.
- High-grade recall: 87.88% ± 9.66 percentage points.

The five test sets contain 93 unique cases. Each eligible case appears in a test set exactly once.
Pooled out-of-fold accuracy is 83.87% (78/93). This weights cases equally; the mean split accuracy weights splits equally.

## Execution and verification

- Seed 3902026; identical baseline training parameters and Python source hashes across all five splits.
- Maximum 30 epochs, batch size 1, 2 data-loader workers, 500 patches per slice, and at most 5 slices per stain. Each job requests 8 CPUs and 32 GB of memory.
- Each run starts a new model. Shared precomputed 4096-dimensional features are used; CNN feature extraction is not repeated.
- Verified test IDs and labels against Stage 3 assignments, probability consistency, recomputed metrics, distinct output directories, checkpoint availability, and Slurm COMPLETED / ExitCode 0:0 for all five runs.
- Main program runtime includes data preparation, training, validation, and testing; it excludes queue time and reporting. Per-job runtimes overlap for parallel jobs.
- Each result uses the final in-memory epoch, preserving the provided baseline behavior. The saved best.pth is not automatically reloaded before testing.

The canceled startup attempt (array 4843638) is excluded from all results; its temporary outputs and failed logs have been removed. Every split now has a separate runs/splitNN/ output parent. Split 01 was relocated after completion, with the original log paths preserved and the relocation recorded in its run manifest. Model and training source files are unchanged.

## Files

- split_metrics.csv: per-split metrics, case counts, epochs, main/Slurm runtimes, job references, and Quest output paths.
- out_of_fold_predictions.csv: predictions and both class probabilities for all 93 test cases, with split membership.
- loss_curves_all_splits.png: training and validation loss for each split.
- summary.json: machine-readable results and verification status.
- slurm_accounting.txt: Slurm completion evidence.

Labels: 0 = benign; 1 = high-grade. Confusion matrices use true classes as rows and predicted classes as columns.

For critical reflection, inspect the loss curves for overfitting and consider loading the checkpoint selected by validation loss before testing in a separate experiment. No checkpoint-selection change was applied to these five baseline runs.
