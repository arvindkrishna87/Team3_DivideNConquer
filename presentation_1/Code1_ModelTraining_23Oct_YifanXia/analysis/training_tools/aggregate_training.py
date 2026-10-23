"""Validate and summarize five completed personal baseline experiments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(base, output):
    assignments = read_csv(base / 'analysis/case_assignments.csv')
    all_cases = {int(r['case_id']) for r in assignments}
    manifests, metrics, histories = [], [], []
    combined, rows, seen_runs = [], [], set()
    reference_arguments = None

    for split in range(1, 6):
        directory = base / 'analysis' / f'split{split:02d}'
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        metric = json.loads((directory / 'metrics.json').read_text())
        if manifest['status'] != 'COMPLETED' or not metric['validation_checks_passed']:
            raise RuntimeError(f'Split {split} is not complete and validated')
        assert manifest['split'] == metric['split'] == split
        assert manifest['seed'] == metric['seed'] == 3902026
        if manifests:
            for key in ('training_config', 'source_sha256', 'package_versions',
                        'python', 'cpu_cores_requested', 'cpu_threads', 'checkpoint_policy'):
                assert manifest[key] == manifests[0][key], (split, key)
            for key in ('labels_csv', 'patches_dir'):
                assert manifest['data_paths'][key] == manifests[0]['data_paths'][key]
        arguments = list(manifest.get('main_arguments', manifest['command'][3:]))
        index = arguments.index('--load_splits')
        split_file = base / 'splits' / f'data_splits_new_{split:02d}.npz'
        assert Path(arguments[index + 1]) == split_file
        assert hashlib.sha256(split_file.read_bytes()).hexdigest() == manifest['split_sha256']
        arguments[index + 1] = '<split_file>'
        if reference_arguments is None:
            reference_arguments = arguments
        assert arguments == reference_arguments
        with np.load(split_file, allow_pickle=False) as data:
            split_ids = {subset: set(map(int, data[subset + '_cases']))
                         for subset in ('train', 'val', 'test')}
        for subset in split_ids:
            expected = {int(r['case_id']) for r in assignments
                        if int(r['split']) == split and r['subset'] == subset}
            assert split_ids[subset] == expected
        assert not (split_ids['train'] & split_ids['val'] or
                    split_ids['train'] & split_ids['test'] or
                    split_ids['val'] & split_ids['test'])

        run = Path(metric['run_directory'])
        assert str(run) == manifest['run_directory'] and run not in seen_runs
        expected_parent = base / 'runs' / f'split{split:02d}'
        assert run.resolve().parent == expected_parent.resolve()
        seen_runs.add(run)
        predictions = read_csv(run / 'predictions.csv')
        expected_labels = {int(r['case_id']): int(r['label']) for r in assignments
                           if int(r['split']) == split and r['subset'] == 'test'}
        assert len(predictions) == len(expected_labels) == metric['n_test_cases']
        assert {int(r['case_id']): int(r['true_label']) for r in predictions} == expected_labels
        matrix = [[0, 0], [0, 0]]
        for prediction in predictions:
            true, pred = int(prediction['true_label']), int(prediction['predicted_label'])
            probabilities = [float(prediction['prob_class0']), float(prediction['prob_class1'])]
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            assert math.isclose(sum(probabilities), 1, abs_tol=1e-5)
            assert pred == int(np.argmax(probabilities))
            assert prediction['correct'].lower() == str(true == pred).lower()
            matrix[true][pred] += 1
            combined.append(dict(split=split, **prediction))
        assert matrix == metric['confusion_matrix_true_rows_predicted_columns']
        assert metric['correct'] == matrix[0][0] + matrix[1][1]
        assert math.isclose(metric['accuracy'], metric['correct'] / len(predictions))
        assert math.isclose(metric['benign_recall'], matrix[0][0] / sum(matrix[0]))
        assert math.isclose(metric['high_grade_recall'], matrix[1][1] / sum(matrix[1]))

        history = read_csv(directory / 'epoch_history.csv')
        assert [int(r['epoch']) for r in history] == list(range(1, metric['epochs_completed'] + 1))
        assert all(math.isfinite(float(row[key])) for row in history for key in row)
        assert (directory / 'loss_curves.png').stat().st_size > 1000
        assert Path(metric['evaluated_checkpoint']).is_file()
        assert (run / 'confusion_matrix.png').stat().st_size > 1000
        job = (manifest['array_job_id'] + '_' + manifest['array_task_id']
               if manifest.get('array_job_id') else manifest['job_id'])
        rows.append(dict(split=split, train_cases=len(split_ids['train']),
                         validation_cases=len(split_ids['val']), test_cases=len(predictions),
                         correct=metric['correct'], accuracy=metric['accuracy'],
                         benign_recall=metric['benign_recall'],
                         high_grade_recall=metric['high_grade_recall'],
                         epochs=metric['epochs_completed'],
                         main_execution_seconds=metric['main_execution_seconds'],
                         job_reference=job, run_directory=str(run)))
        manifests.append(manifest)
        metrics.append(metric)
        histories.append(history)

    ids = [int(row['case_id']) for row in combined]
    assert len(ids) == len(set(ids)) == len(all_cases) == 93
    assert set(ids) == all_cases
    for name, digest in manifests[0]['source_sha256'].items():
        assert hashlib.sha256((base / 'code' / name).read_bytes()).hexdigest() == digest
    references = {row['job_reference'] for row in rows}
    accounting = subprocess.check_output(
        ['sacct', '-n', '-P', '-j', ','.join(sorted(references)),
         '--format=JobID,State,ExitCode,ElapsedRaw'], text=True)
    states = {}
    for line in accounting.splitlines():
        fields = line.split('|')
        if fields[0] in references:
            states[fields[0]] = fields[1:4]
    assert set(states) == references, states
    for row in rows:
        state, exitcode, elapsed = states[row['job_reference']]
        if state != 'COMPLETED' or exitcode != '0:0':
            raise RuntimeError(f"Slurm has not confirmed completion: {row['job_reference']} {state} {exitcode}")
        row['slurm_execution_seconds'] = int(elapsed)

    statistics = {}
    for key in ('accuracy', 'benign_recall', 'high_grade_recall', 'main_execution_seconds'):
        values = [row[key] for row in rows]
        statistics[key] = dict(mean=float(np.mean(values)),
                               sample_standard_deviation=float(np.std(values, ddof=1)),
                               minimum=min(values), maximum=max(values))
    pooled_matrix = np.sum([m['confusion_matrix_true_rows_predicted_columns'] for m in metrics], axis=0)
    pooled = dict(n_cases=len(ids), correct=int(np.trace(pooled_matrix)),
                  accuracy=float(np.trace(pooled_matrix) / len(ids)),
                  benign_recall=float(pooled_matrix[0, 0] / pooled_matrix[0].sum()),
                  high_grade_recall=float(pooled_matrix[1, 1] / pooled_matrix[1].sum()),
                  confusion_matrix_true_rows_predicted_columns=pooled_matrix.tolist())
    summary = dict(status='COMPLETED', seed=3902026, splits_completed=5,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   split_metrics=rows, across_split_statistics=statistics,
                   pooled_out_of_fold=pooled,
                   standard_deviation_note='Sample SD across five splits (ddof=1); descriptive variability, not a confidence interval.',
                   runtime_note='Main runtime includes data preparation, training, validation and testing; excludes queue and report generation. Runtimes are per job and overlap for parallel runs.',
                   checkpoint_policy='Each split evaluates its final in-memory epoch, matching the provided baseline; best.pth is not reloaded.',
                   excluded_attempt='Array 4843638 was canceled after an output-directory collision and excluded from all reported results. Its temporary outputs and failed logs were removed during workspace cleanup.',
                   validation_checks_passed=True)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'split_metrics.csv', rows)
    write_csv(output / 'out_of_fold_predictions.csv', sorted(combined, key=lambda r: int(r['case_id'])))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'slurm_accounting.txt').write_text(accounting)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for index, (history, row) in enumerate(zip(histories, rows)):
        ax = axes.flat[index]
        epochs = [int(r['epoch']) for r in history]
        ax.plot(epochs, [float(r['train_ce_loss']) for r in history], color='#2265a5', label='Train cross-entropy')
        ax.plot(epochs, [float(r['val_ce_loss']) for r in history], color='#d97525', label='Validation cross-entropy')
        ax.set(title=f"Split {row['split']:02d} ({row['epochs']} epochs)", xlabel='Epoch', ylabel='Loss')
        ax.grid(alpha=.2)
    axes.flat[5].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc='upper left', frameon=False)
    axes.flat[5].text(.02, .60, 'Seed: 3902026\nMaximum: 30 epochs\nExisting early stopping enabled\n\nTested model: final epoch in each split\nLoss values: baseline logs (4 decimals)',
                      transform=axes.flat[5].transAxes, va='top', fontsize=11, linespacing=1.5)
    fig.suptitle('Training and validation loss across five splits', fontsize=17)
    fig.savefig(output / 'loss_curves_all_splits.png', dpi=180)
    plt.close(fig)

    lines = ['# Yifan Xia — Stage 4: Five-Split Results', '',
             'All five baseline training, validation, and test runs completed successfully on Quest.', '',
             '| Split | Train / Val / Test | Accuracy | Benign recall | High-grade recall | Epochs | Main runtime (s) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['split']:02d} | {row['train_cases']} / {row['validation_cases']} / {row['test_cases']} | {row['accuracy']:.2%} | {row['benign_recall']:.2%} | {row['high_grade_recall']:.2%} | {row['epochs']} | {row['main_execution_seconds']:.2f} |")
    lines += ['', '## Across-split summary', '',
              'Mean ± sample SD across five splits (percentage points for SD):', '']
    for key, label in [('accuracy', 'Accuracy'), ('benign_recall', 'Benign recall'), ('high_grade_recall', 'High-grade recall')]:
        stat = statistics[key]
        lines.append(f"- {label}: {stat['mean']:.2%} ± {100 * stat['sample_standard_deviation']:.2f} percentage points.")
    lines += ['', 'The five test sets contain 93 unique cases. Each eligible case appears in a test set exactly once.',
              f"Pooled out-of-fold accuracy is {pooled['accuracy']:.2%} ({pooled['correct']}/93). This weights cases equally; the mean split accuracy weights splits equally.", '',
              '## Execution and verification', '',
              '- Seed 3902026; identical baseline training parameters and Python source hashes across all five splits.',
              '- Maximum 30 epochs, batch size 1, 2 data-loader workers, 500 patches per slice, and at most 5 slices per stain. Each job requests 8 CPUs and 32 GB of memory.',
              '- Each run starts a new model. Shared precomputed 4096-dimensional features are used; CNN feature extraction is not repeated.',
              '- Verified test IDs and labels against Stage 3 assignments, probability consistency, recomputed metrics, distinct output directories, checkpoint availability, and Slurm COMPLETED / ExitCode 0:0 for all five runs.',
              '- Main program runtime includes data preparation, training, validation, and testing; it excludes queue time and reporting. Per-job runtimes overlap for parallel jobs.',
              '- Each result uses the final in-memory epoch, preserving the provided baseline behavior. The saved best.pth is not automatically reloaded before testing.', '',
              'The canceled startup attempt (array 4843638) is excluded from all results; its temporary outputs and failed logs have been removed. Every split now has a separate runs/splitNN/ output parent. Split 01 was relocated after completion, with the original log paths preserved and the relocation recorded in its run manifest. Model and training source files are unchanged.', '',
              '## Files', '',
              '- split_metrics.csv: per-split metrics, case counts, epochs, main/Slurm runtimes, job references, and Quest output paths.',
              '- out_of_fold_predictions.csv: predictions and both class probabilities for all 93 test cases, with split membership.',
              '- loss_curves_all_splits.png: training and validation loss for each split.',
              '- summary.json: machine-readable results and verification status.',
              '- slurm_accounting.txt: Slurm completion evidence.', '',
              'Labels: 0 = benign; 1 = high-grade. Confusion matrices use true classes as rows and predicted classes as columns.', '',
              'For critical reflection, inspect the loss curves for overfitting and consider loading the checkpoint selected by validation loss before testing in a separate experiment. No checkpoint-selection change was applied to these five baseline runs.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.workspace, args.output)
