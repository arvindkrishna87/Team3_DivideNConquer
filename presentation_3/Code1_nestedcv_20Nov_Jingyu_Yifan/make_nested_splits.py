#!/usr/bin/env python3
"""
Build the five case groups used by nested cross-validation.

This does not modify make_splits.py and does not write the old 3+1+1
train/val/test files. It stops at five groups, saved as

    nested_splits/nested_groups.npz

with keys group_A ... group_E.

The constraints match make_splits.py: split by case, keep GROUPED_CASES
together, and shuffle group ids with the training seed before
StratifiedGroupKFold(n_splits=5).
"""

import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_PATHS, GROUPED_CASES, TRAINING_CONFIG
from data_utils import (
    build_slice_to_class_map,
    get_all_patch_files,
    group_patches_by_slice,
    load_labels,
)

GROUP_NAMES = ("A", "B", "C", "D", "E")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def case_labels_from_slices(slice_to_class: dict) -> Dict[int, int]:
    """One label per case. A case with conflicting slice labels is an error."""
    found: Dict[int, set] = {}
    for (case_id, _slice_id), y in slice_to_class.items():
        found.setdefault(int(case_id), set()).add(int(y))
    case_to_label = {}
    for case_id, labels in found.items():
        if len(labels) != 1:
            raise RuntimeError(f"Case {case_id} has conflicting labels: {sorted(labels)}")
        case_to_label[case_id] = next(iter(labels))
    return case_to_label


def assign_five_groups(
    case_to_label: Dict[int, int],
    grouped_cases: Sequence[Sequence[int]],
    seed: int,
) -> Dict[str, List[int]]:
    """
    Return {A..E: sorted case ids}.

    Grouped cases share one StratifiedGroupKFold group id (the first id in
    the tuple). Group ids are then relabeled with RandomState(seed), and
    the fold order of StratifiedGroupKFold(n_splits=5) becomes A..E.
    """
    if len(GROUP_NAMES) != 5:
        raise RuntimeError("Expected exactly five group names.")

    case_ids = np.array(sorted(case_to_label), dtype=int)
    if case_ids.size == 0:
        raise RuntimeError("No cases found to split.")

    labels = np.array([case_to_label[int(c)] for c in case_ids], dtype=int)
    n_benign = int(np.sum(labels == 0))
    n_high = int(np.sum(labels == 1))
    if n_benign == 0 or n_high == 0:
        raise RuntimeError(
            f"Both classes are required to stratify. benign={n_benign}, high={n_high}"
        )

    groups = case_ids.copy()
    for bundle in grouped_cases:
        mask = np.isin(groups, list(bundle))
        if mask.any():
            np.place(groups, mask, int(bundle[0]))

    unique_ids = np.unique(groups)
    relabeled = dict(
        zip(
            unique_ids,
            np.random.RandomState(seed).permutation(len(unique_ids)),
        )
    )
    groups = np.array([relabeled[g] for g in groups])

    splitter = StratifiedGroupKFold(n_splits=5)
    folds: List[List[int]] = []
    for _, test_index in splitter.split(case_ids, labels, groups):
        folds.append(sorted(int(case_ids[i]) for i in test_index))

    if len(folds) != 5:
        raise RuntimeError(f"StratifiedGroupKFold returned {len(folds)} folds, expected 5.")

    return {name: fold for name, fold in zip(GROUP_NAMES, folds)}


def validate_groups(
    groups: Dict[str, List[int]],
    case_to_label: Dict[int, int],
    grouped_cases: Sequence[Sequence[int]],
) -> None:
    """Raise if the five groups are not a usable nested-CV partition."""
    names = list(GROUP_NAMES)
    if list(groups) != names and set(groups) != set(names):
        raise RuntimeError(f"Groups must be named A-E, got {sorted(groups)}")

    seen = []
    for name in names:
        cases = groups[name]
        if len(cases) != len(set(cases)):
            raise RuntimeError(f"Group {name} contains duplicate case ids.")
        labels = [case_to_label[c] for c in cases]
        n_benign = sum(y == 0 for y in labels)
        n_high = sum(y == 1 for y in labels)
        if n_benign == 0 or n_high == 0:
            raise RuntimeError(
                f"Group {name} is missing a class (benign={n_benign}, high={n_high}). "
                "Validation loss needs both classes."
            )
        seen.extend(cases)

    if len(seen) != len(set(seen)):
        raise RuntimeError("A case appears in more than one group.")
    missing = sorted(set(case_to_label) - set(seen))
    extra = sorted(set(seen) - set(case_to_label))
    if missing or extra:
        raise RuntimeError(
            "Groups do not match the labeled cases with patches. "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    for bundle in grouped_cases:
        present = [c for c in bundle if c in case_to_label]
        if len(present) <= 1:
            continue
        homes = set()
        for name in names:
            members = set(groups[name])
            if any(c in members for c in present):
                homes.add(name)
        if len(homes) > 1:
            raise RuntimeError(
                f"Grouped cases {tuple(present)} were split across {sorted(homes)}."
            )


def summary_lines(groups: Dict[str, List[int]], case_to_label: Dict[int, int]) -> List[str]:
    lines = []
    n_benign = sum(y == 0 for y in case_to_label.values())
    n_high = sum(y == 1 for y in case_to_label.values())
    lines.append(f"Total cases: {len(case_to_label)}  |  Benign: {n_benign}  |  High-grade: {n_high}")
    lines.append("")
    for name in GROUP_NAMES:
        cases = groups[name]
        benign = [c for c in cases if case_to_label[c] == 0]
        high = [c for c in cases if case_to_label[c] == 1]
        lines.append("=" * 40)
        lines.append(f"GROUP {name}")
        lines.append("=" * 40)
        lines.append(f"Total cases:    {len(cases)}")
        lines.append(f"Benign (0):     {len(benign)}")
        lines.append(f"High-grade (1): {len(high)}")
        lines.append(f"Benign CASE IDs: {benign}")
        lines.append(f"High-grade CASE IDs: {high}")
        lines.append("")
    return lines


def save_groups(groups: Dict[str, List[int]], save_dir: str, case_to_label: Dict[int, int]) -> str:
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, "nested_groups.npz")
    payload = {f"group_{name}": np.array(groups[name], dtype=np.int64) for name in GROUP_NAMES}
    np.savez(npz_path, **payload)

    text_path = os.path.join(save_dir, "nested_groups_summary.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines(groups, case_to_label)))
        handle.write("\n")
    return npz_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the five nested-CV case groups.")
    parser.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
    parser.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
    parser.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"])
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "nested_splits"),
    )
    parser.add_argument(
        "--grouped_cases",
        type=str,
        default=str(list(GROUPED_CASES)),
        help="Python literal of tuples, same format as make_splits.py.",
    )
    return parser.parse_args()


def main() -> None:
    import ast

    args = parse_args()
    grouped_cases = ast.literal_eval(args.grouped_cases)

    print("=" * 80)
    print("NESTED CV GROUPS (FIVE GROUPS, NO 3+1+1 PACKING)")
    print("=" * 80)
    print(f"labels_csv:    {args.labels_csv}")
    print(f"patches_dir:   {args.patches_dir}")
    print(f"seed:          {args.seed}")
    print(f"save_dir:      {args.save_dir}")
    print(f"grouped_cases: {grouped_cases}")

    labels = load_labels(args.labels_csv)
    print(f"\nLoaded {len(labels)} label rows")
    all_files = get_all_patch_files(args.patches_dir)
    print(f"Found {len(all_files)} files in patches_dir")
    patches = group_patches_by_slice(all_files, args.patches_dir)
    print(f"Grouped into {len(patches)} slices")
    slice_to_class = build_slice_to_class_map(patches, labels)
    print(f"Mapped {len(slice_to_class)} slices to classes")

    case_to_label = case_labels_from_slices(slice_to_class)
    groups = assign_five_groups(case_to_label, grouped_cases, args.seed)
    validate_groups(groups, case_to_label, grouped_cases)

    for line in summary_lines(groups, case_to_label):
        print(line)

    npz_path = save_groups(groups, args.save_dir, case_to_label)
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()
