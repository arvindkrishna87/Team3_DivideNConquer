"""Summarize one completed baseline run; do not train or load model weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def summarize(base: Path, split: int, log_path: Path, output: Path):
    text = log_path.read_text(errors='replace')
    if 'Training completed successfully!' not in text:
        raise RuntimeError('main.py did not report successful completion')
    matches = re.findall(r'Created run directory:\s*([^\r\n]+)', text)
    if len(matches) != 1:
        raise RuntimeError('Expected one run directory in the log')
    run = Path(matches[0].strip())
    manifest_path = base / 'analysis' / f'split{split:02d}' / 'run_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        relocation = manifest.get('output_relocation', {})
        if str(run) == relocation.get('original_run_directory'):
            run = Path(relocation['current_run_directory'])
    expected_parent = base / 'runs' / f'split{split:02d}'
    if run.resolve().parent != expected_parent.resolve():
        raise RuntimeError(f'Unexpected run directory: {run}')

    with (run / 'predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    with np.load(base / 'splits' / f'data_splits_new_{split:02d}.npz',
                 allow_pickle=False) as saved:
        expected_ids = {int(c) for c in saved['test_cases']}
    actual_ids = [int(row['case_id']) for row in predictions]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise RuntimeError('Prediction IDs do not exactly match the intended test set')

    matrix = [[0, 0], [0, 0]]
    for row in predictions:
        true, pred = int(row['true_label']), int(row['predicted_label'])
        if true not in (0, 1) or pred not in (0, 1):
            raise RuntimeError('Unexpected prediction label')
        probabilities = [float(row['prob_class0']), float(row['prob_class1'])]
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise RuntimeError('Invalid prediction probability')
        if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
            raise RuntimeError('Prediction probabilities do not sum to one')
        if pred != int(np.argmax(probabilities)):
            raise RuntimeError('Prediction label does not match argmax probability')
        if row['correct'].lower() != str(true == pred).lower():
            raise RuntimeError('Prediction correct flag is inconsistent')
        matrix[true][pred] += 1

    n = len(predictions)
    correct = matrix[0][0] + matrix[1][1]
    accuracy = correct / n
    recalls = [matrix[c][c] / sum(matrix[c]) if sum(matrix[c]) else None
               for c in (0, 1)]
    results = (run / 'results.txt').read_text()
    reported_n = int(re.search(r'Number of samples:\s*(\d+)', results).group(1))
    reported_acc = float(re.search(r'Test Accuracy:\s*([\d.eE+-]+)', results).group(1))
    if reported_n != n or abs(reported_acc - accuracy) > 5.1e-5:
        raise RuntimeError('results.txt disagrees with predictions.csv')

    number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    epoch_pattern = re.compile(r'^Epoch (\d+)/(\d+) \(LR: ' + number + r'\)$')
    losses_pattern = re.compile(
        r'^Train Loss: ' + number + r' \| Train CE: ' + number +
        r' \| Train Patch Entropy: ' + number + r' \| Val Loss: ' + number +
        r' \| Val Acc: ' + number + r'$')
    history, active = [], None
    for raw in text.splitlines():
        line = raw.strip()
        match = epoch_pattern.match(line)
        if match:
            active = {'epoch': int(match.group(1)),
                      'max_epochs': int(match.group(2)), 'learning_rate': float(match.group(3))}
            continue
        match = losses_pattern.match(line)
        if match and active is not None:
            for name, value in zip(('train_total_loss', 'train_ce_loss',
                                    'train_patch_entropy', 'val_ce_loss', 'val_accuracy'),
                                   match.groups()):
                active[name] = float(value)
            history.append(active)
            active = None
        match = re.match(r'^Epoch Time: ' + number + r's$', line)
        if match and history:
            history[-1]['epoch_seconds'] = float(match.group(1))
    if not history or [r['epoch'] for r in history] != list(range(1, len(history) + 1)):
        raise RuntimeError('Epoch history is empty or incomplete')
    if not all(math.isfinite(v) for row in history for v in row.values()):
        raise RuntimeError('Non-finite value in training history')
    elapsed_match = re.search(r'Total execution time:\s*' + number + r's', text)
    if not elapsed_match:
        raise RuntimeError('Missing main.py runtime')
    main_seconds = float(elapsed_match.group(1))
    last_epoch = history[-1]['epoch']
    checkpoints = sorted((run / 'checkpoints').glob(f'*_epoch{last_epoch}.pth'))
    if len(checkpoints) != 1:
        raise RuntimeError('Cannot identify the final-epoch checkpoint')
    if not (run / 'confusion_matrix.png').is_file():
        raise RuntimeError('Missing baseline confusion matrix')

    output.mkdir(parents=True, exist_ok=True)
    fields = ['epoch', 'max_epochs', 'learning_rate', 'train_total_loss', 'train_ce_loss',
              'train_patch_entropy', 'val_ce_loss', 'val_accuracy', 'epoch_seconds']
    with (output / 'epoch_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    epochs = [r['epoch'] for r in history]
    ax.plot(epochs, [r['train_ce_loss'] for r in history], label='Train cross-entropy',
            color='#2265a5', linewidth=2)
    ax.plot(epochs, [r['val_ce_loss'] for r in history], label='Validation cross-entropy',
            color='#d97525', linewidth=2)
    ax.plot(epochs, [r['train_total_loss'] for r in history],
            label='Train total (CE minus entropy regularization)', color='#4e6656',
            linewidth=1.2, linestyle='--', alpha=.8)
    ax.set(title=f'Split {split:02d}: training and validation loss',
           xlabel='Epoch', ylabel='Loss (mean over cases)')
    ax.set_xlim(.5, max(epochs) + .5)
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(output / 'loss_curves.png', dpi=180)
    plt.close(fig)

    metrics = {
        'split': split, 'seed': 3902026, 'run_directory': str(run),
        'n_test_cases': n, 'correct': correct, 'accuracy': accuracy,
        'benign_recall': recalls[0], 'high_grade_recall': recalls[1],
        'confusion_matrix_true_rows_predicted_columns': matrix,
        'class_order': ['benign', 'high_grade'],
        'test_loss_reported': float(re.search(r'Test Loss:\s*([\d.eE+-]+)', results).group(1)),
        'epochs_completed': last_epoch,
        'main_execution_seconds': main_seconds,
        'main_runtime_scope': 'Data preparation, training, validation and testing; excludes queue and reporting.',
        'epoch_seconds_sum': sum(r.get('epoch_seconds', 0) for r in history),
        'checkpoint_policy': 'Last in-memory epoch, matching provided main.py; best.pth is not reloaded.',
        'evaluated_checkpoint': str(checkpoints[0]),
        'predictions_csv': str(run / 'predictions.csv'),
        'confusion_matrix_png': str(run / 'confusion_matrix.png'),
        'epoch_history_precision': 'Losses parsed from the baseline four-decimal stdout records.',
        'validation_checks_passed': True,
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.workspace, args.split, args.log, args.output)
