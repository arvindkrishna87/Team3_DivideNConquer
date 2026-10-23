"""Validate Yifan's Stage 3 case-grouped splits without training any model."""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--reference-splits', type=Path, required=True)
    args = parser.parse_args()
    base = args.workspace
    sys.path.insert(0, str(base / 'code'))
    from config import DATA_PATHS, GROUPED_CASES
    from data_utils import (load_labels, get_all_patch_files,
                            group_patches_by_slice, build_slice_to_class_map)
    from make_splits import split_by_case_with_constraints

    with contextlib.redirect_stdout(io.StringIO()):
        labels = load_labels(DATA_PATHS['labels_csv'])
        paths = group_patches_by_slice(
            get_all_patch_files(DATA_PATHS['patches_dir']), DATA_PATHS['patches_dir'])
        slice_labels = build_slice_to_class_map(paths, labels)
    case_labels = {}
    for (case_id, _), label in slice_labels.items():
        cid, y = int(case_id), int(label)
        if cid in case_labels and case_labels[cid] != y:
            raise RuntimeError(f'Conflicting labels for case {cid}')
        case_labels[cid] = y
    universe = set(case_labels)
    group_ids = {cid: cid for cid in universe}
    for family in GROUPED_CASES:
        for cid in family:
            if cid in group_ids:
                group_ids[cid] = int(family[0])

    checks = []

    def check(name, condition, detail=None):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    def read_split(path):
        with np.load(path, allow_pickle=False) as content:
            return {key: sorted(int(c) for c in content[key])
                    for key in ('train_cases', 'val_cases', 'test_cases')}

    def reconstruct(seed):
        with contextlib.redirect_stdout(io.StringIO()):
            _, ratios, folds = split_by_case_with_constraints(
                slice_labels, GROUPED_CASES, seed, 0.6, 0.2, 0.2, 20)
        tr, va, te = ratios
        records = []
        for _ in range(len(folds) // te):
            records.append({
                'train_cases': sorted(c for f in folds[:tr] for c in f),
                'val_cases': sorted(c for f in folds[tr:tr + va] for c in f),
                'test_cases': sorted(c for f in folds[-te:] for c in f),
            })
            folds = folds[te:] + folds[:te]
        return records

    split_paths = sorted((base / 'splits').glob('data_splits_new_*.npz'))
    check('exactly_five_split_files', len(split_paths) == 5, len(split_paths))
    actual = [read_split(p) for p in split_paths]
    summary, assignments = [], []
    test_counts = collections.Counter()
    overall_high = sum(case_labels.values()) / len(universe)
    max_deviation = 0.15

    for index, data in enumerate(actual, 1):
        sets = {k: set(v) for k, v in data.items()}
        check(f'split_{index}_no_duplicate_case_ids',
              all(len(data[k]) == len(sets[k]) for k in data))
        check(f'split_{index}_covers_all_eligible_cases',
              set.union(*sets.values()) == universe)
        keys = list(sets)
        check(f'split_{index}_disjoint_case_ids',
              all(not (sets[keys[i]] & sets[keys[j]])
                  for i in range(3) for j in range(i + 1, 3)))
        group_sets = [{group_ids[c] for c in sets[key]} for key in keys]
        check(f'split_{index}_disjoint_case_families',
              all(not (group_sets[i] & group_sets[j])
                  for i in range(3) for j in range(i + 1, 3)))
        for key, cases in data.items():
            counts = collections.Counter(case_labels[c] for c in cases)
            high_ratio = counts[1] / len(cases) if cases else None
            check(f'split_{index}_{key}_both_classes', counts[0] > 0 and counts[1] > 0)
            deviation = abs(high_ratio - overall_high) if high_ratio is not None else 1.0
            check(f'split_{index}_{key}_class_balance', deviation <= max_deviation,
                  {'high_grade_fraction': high_ratio,
                   'overall_high_grade_fraction': overall_high,
                   'maximum_absolute_deviation': max_deviation})
            summary.append({'split': index, 'subset': key.replace('_cases', ''),
                            'n_cases': len(cases), 'benign': counts[0],
                            'high_grade': counts[1], 'high_grade_fraction': high_ratio})
            assignments.extend({'split': index, 'subset': key.replace('_cases', ''),
                                'case_id': cid, 'label': case_labels[cid],
                                'family_id': group_ids[cid]} for cid in cases)
        test_counts.update(data['test_cases'])

    check('every_eligible_case_tested_exactly_once',
          set(test_counts) == universe and all(n == 1 for n in test_counts.values()))
    repeated = reconstruct(args.seed)
    check('same_seed_reproduces_saved_splits', repeated == actual)
    alternate = reconstruct(args.seed + 1)
    own_test_partition = {frozenset(d['test_cases']) for d in actual}
    check('different_seed_changes_test_partition',
          own_test_partition != {frozenset(d['test_cases']) for d in alternate})

    reference_paths = sorted(args.reference_splits.glob('data_splits_new_*.npz'))
    reference = [read_split(p) for p in reference_paths]
    check('five_reference_splits_available', len(reference) == 5)
    check('test_fold_partition_differs_from_reference',
          bool(reference) and own_test_partition !=
          {frozenset(d['test_cases']) for d in reference})
    for index, data in enumerate(actual, 1):
        check(f'split_{index}_not_duplicate_of_any_reference_assignment',
              bool(reference) and all(data != other for other in reference))

    out = base / 'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in [('split_summary.csv', summary), ('case_assignments.csv', assignments)]:
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed, 'eligible_cases': len(universe),
        'benign_cases': sum(y == 0 for y in case_labels.values()),
        'high_grade_cases': sum(case_labels.values()),
        'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
        'reference_splits': str(args.reference_splits),
        'balance_check_rule': 'Absolute class-fraction deviation <= 0.15; no model predictions used.',
        'all_checks_passed': all(item['passed'] for item in checks),
        'checks': checks,
        'splits': summary,
        'source_hashes': {
            name: hashlib.sha256((base / 'code' / name).read_bytes()).hexdigest()
            for name in ('make_splits.py', 'data_utils.py', 'config.py')},
        'labels_sha256': hashlib.sha256(Path(DATA_PATHS['labels_csv']).read_bytes()).hexdigest(),
    }
    (out / 'split_validation.json').write_text(json.dumps(report, indent=2))
    lines = ['# Stage 3 split validation', '', f'Seed: {args.seed}',
             f'Eligible cases: {len(universe)}',
             f'All checks passed: {report["all_checks_passed"]}', '',
             '| Split | Subset | Cases | Benign | High-grade |',
             '|---|---|---:|---:|---:|']
    lines.extend(f'| {r["split"]} | {r["subset"]} | {r["n_cases"]} | '
                 f'{r["benign"]} | {r["high_grade"]} |' for r in summary)
    lines += ['', 'Checks:', '']
    lines.extend(f'- {"PASS" if c["passed"] else "FAIL"}: {c["check"]}' for c in checks)
    (out / 'split_validation.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'all_checks_passed': report['all_checks_passed'],
                      'checks': len(checks), 'eligible_cases': len(universe),
                      'seed': args.seed, 'summary': summary}, indent=2))
    failed = [c['check'] for c in checks if not c['passed']]
    if failed:
        raise RuntimeError('Split validation failed: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
