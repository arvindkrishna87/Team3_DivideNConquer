#!/usr/bin/env python3
"""
Nested cross-validation for the hierarchical attention MIL model.

For each outer test group (A-E) this script runs its own Optuna study on the
other four groups. One trial trains four models in lockstep. After every
epoch the four balanced validation losses are averaged, and the trial score
is the lowest of those averages. Refit fixes those hyperparameters and that
epoch count before training. Each refit epoch records training loss, then
balanced test loss on the held-out group with no gradient. That test loss does
not change the learning rate, the number of epochs, or the saved checkpoint.
Accuracy and both recalls are reported only at the predetermined epoch.

Groups come from nested_splits/nested_groups.npz (see make_nested_splits.py).
"""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader

from config import DATA_PATHS, MODEL_CONFIG, TRAINING_CONFIG
from data_utils import (
    build_case_dict,
    build_slice_to_class_map,
    get_all_patch_files,
    group_patches_by_slice,
    load_labels,
)
from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
from models import HierarchicalAttnMIL
from utils import get_device, set_seed

GROUP_NAMES = ("A", "B", "C", "D", "E")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Scientific ranges. Zero for weight decay and entropy is a separate choice.
DEFAULT_BOUNDS = {
    "learning_rate": (1e-5, 1e-3),
    "weight_decay": (1e-5, 1e3),
    "entropy_lambda": (1e-3, 1.0),
    "classifier_dropout": (0.0, 0.5),
    "projector_dropout": (0.0, 0.5),
}
LOG_KEYS = ("learning_rate", "weight_decay", "entropy_lambda")
DROPOUT_KEYS = ("classifier_dropout", "projector_dropout")
EDGE_FRACTION = 0.05
PARAM_ORDER = (
    "learning_rate",
    "weight_decay",
    "classifier_dropout",
    "entropy_lambda",
    "projector_dropout",
)


def good_set_size(n: int) -> int:
    """Optuna TPE good-set size: max(6, ceil(0.15 n)), n = completed trials."""
    n = int(n)
    return max(6, math.ceil(0.15 * n))


def class_weights_from_labels(labels: Sequence[int]) -> Tuple[float, float, int, int]:
    """w_c = N / (2 n_c) from the current training cases. Returns (w_b, w_h, n_b, n_h)."""
    labels = [int(y) for y in labels]
    n = len(labels)
    n_b = sum(y == 0 for y in labels)
    n_h = sum(y == 1 for y in labels)
    if n == 0 or n_b == 0 or n_h == 0:
        raise RuntimeError(
            f"Training split needs both classes to weight CE (n={n}, benign={n_b}, high={n_h})."
        )
    return n / (2.0 * n_b), n / (2.0 * n_h), n_b, n_h


def balanced_validation_loss(sum_benign: float, n_benign: int, sum_high: float, n_high: int) -> float:
    """Mean of the two class-wise mean CEs. No class weights and no entropy."""
    if n_benign <= 0 or n_high <= 0:
        raise RuntimeError(
            f"Balanced validation loss needs both classes (benign={n_benign}, high={n_high})."
        )
    return 0.5 * (sum_benign / n_benign + sum_high / n_high)


def balanced_loader_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Balanced CE on a loader. No gradient, no class weights, and no entropy."""
    was_training = model.training
    model.eval()
    sums = {0: 0.0, 1: 0.0}
    counts = {0: 0, 1: 0}
    with torch.inference_mode():
        for batch in loader:
            if not batch:
                continue
            logits, label, _attention = _forward_case(model, batch[0], device, False)
            ce_loss = F.cross_entropy(logits, label, reduction="sum")
            y = int(label.item())
            sums[y] += float(ce_loss.item())
            counts[y] += 1
    if was_training:
        model.train()
    return balanced_validation_loss(sums[0], counts[0], sums[1], counts[1])


def best_epoch_from_curve(mean_losses: Sequence[float]) -> Tuple[int, float]:
    """1-based epoch of the strict minimum. The earliest epoch wins a tie."""
    if len(mean_losses) == 0:
        raise ValueError("Cannot select an epoch from an empty curve.")
    best_index = 0
    best_value = float(mean_losses[0])
    for index, value in enumerate(mean_losses):
        value = float(value)
        if value < best_value:
            best_value = value
            best_index = index
    return best_index + 1, best_value


def _log_position(value: float, low: float, high: float) -> float:
    span = math.log(high) - math.log(low)
    if span <= 0:
        raise ValueError(f"Log range must be increasing, got {(low, high)}")
    return (math.log(value) - math.log(low)) / span


def assess_edges(decoded: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Widen a bound only when values outside the current range are meaningful.

    Dropout near 0 is recorded and does not trigger a rerun. Dropout above
    0.48 widens the upper bound to 0.7. A log-parameter within the outer 5%
    of its log interval widens that end by 10x. Exact zero is a legal point.
    """
    updated = {key: (float(low), float(high)) for key, (low, high) in bounds.items()}
    notes: List[str] = []
    changed = False

    for key in LOG_KEYS:
        value = float(decoded[key])
        if key in ("weight_decay", "entropy_lambda") and value == 0.0:
            notes.append(f"{key}=0 is an allowed search point, not an edge")
            continue
        low, high = updated[key]
        if value <= 0.0:
            raise ValueError(f"{key} must be positive to sit on a log range, got {value}")
        position = _log_position(value, low, high)
        if position <= EDGE_FRACTION:
            new_low = low / 10.0
            updated[key] = (new_low, high)
            changed = True
            notes.append(
                f"{key}={value:.6g} is within the lower {EDGE_FRACTION:.0%} of [{low:.6g}, {high:.6g}]; "
                f"lower bound becomes {new_low:.6g}"
            )
        elif position >= 1.0 - EDGE_FRACTION:
            new_high = high * 10.0
            updated[key] = (low, new_high)
            changed = True
            notes.append(
                f"{key}={value:.6g} is within the upper {EDGE_FRACTION:.0%} of [{low:.6g}, {high:.6g}]; "
                f"upper bound becomes {new_high:.6g}"
            )

    for key in DROPOUT_KEYS:
        value = float(decoded[key])
        low, high = updated[key]
        if value > 0.48 and high < 0.7 - 1e-12:
            updated[key] = (low, 0.7)
            changed = True
            notes.append(f"{key}={value:.6g} is near the upper edge; upper bound becomes 0.7")
        elif value > 0.48:
            notes.append(f"{key}={value:.6g} is near the upper edge, and the upper bound is already {high}")
        elif value < 0.02:
            notes.append(
                f"{key}={value:.6g} is near the legal lower bound 0; the range is not widened"
            )

    return {"changed": changed, "bounds": updated, "notes": notes}


def _map_linear(unit: float, low: float, high: float) -> float:
    unit = min(max(float(unit), 0.0), 1.0)
    return float(low + unit * (high - low))


def _map_log(unit: float, low: float, high: float) -> float:
    unit = min(max(float(unit), 0.0), 1.0)
    return float(math.exp(math.log(low) + unit * (math.log(high) - math.log(low))))


def _decode_spike(unit: float, low: float, high: float) -> Tuple[int, float]:
    """Separate zero from the log range. u < 0.5 selects zero."""
    unit = float(unit)
    if unit < 0.5:
        magnitude = _map_log(unit / 0.5, low, high)
        return 1, magnitude
    magnitude = _map_log((unit - 0.5) / 0.5, low, high)
    return 0, magnitude


def sobol_startup_points(n_points: int, seed: int, bounds: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
    """16 scrambled Sobol points in the five scientific parameters."""
    engine = SobolEngine(dimension=len(PARAM_ORDER), scramble=True, seed=seed)
    draw = engine.draw(n_points).cpu().numpy()
    points = []
    for row in draw:
        wd_zero, wd_mag = _decode_spike(row[1], *bounds["weight_decay"])
        ent_zero, ent_mag = _decode_spike(row[3], *bounds["entropy_lambda"])
        points.append(
            {
                "learning_rate": _map_log(row[0], *bounds["learning_rate"]),
                "weight_decay_is_zero": wd_zero,
                "weight_decay": wd_mag,
                "classifier_dropout": _map_linear(row[2], *bounds["classifier_dropout"]),
                "entropy_lambda_is_zero": ent_zero,
                "entropy_lambda": ent_mag,
                "projector_dropout": _map_linear(row[4], *bounds["projector_dropout"]),
            }
        )
    return points


def suggest_hparams(trial: optuna.Trial, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    lr_low, lr_high = bounds["learning_rate"]
    wd_low, wd_high = bounds["weight_decay"]
    ent_low, ent_high = bounds["entropy_lambda"]
    clf_low, clf_high = bounds["classifier_dropout"]
    proj_low, proj_high = bounds["projector_dropout"]
    learning_rate = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
    weight_decay_is_zero = trial.suggest_categorical("weight_decay_is_zero", [0, 1])
    weight_decay_mag = trial.suggest_float("weight_decay", wd_low, wd_high, log=True)
    classifier_dropout = trial.suggest_float("classifier_dropout", clf_low, clf_high)
    entropy_is_zero = trial.suggest_categorical("entropy_lambda_is_zero", [0, 1])
    entropy_mag = trial.suggest_float("entropy_lambda", ent_low, ent_high, log=True)
    projector_dropout = trial.suggest_float("projector_dropout", proj_low, proj_high)
    return decode_hparams(
        {
            "learning_rate": learning_rate,
            "weight_decay_is_zero": weight_decay_is_zero,
            "weight_decay": weight_decay_mag,
            "classifier_dropout": classifier_dropout,
            "entropy_lambda_is_zero": entropy_is_zero,
            "entropy_lambda": entropy_mag,
            "projector_dropout": projector_dropout,
        }
    )


def decode_hparams(params: Dict[str, Any]) -> Dict[str, float]:
    return {
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": 0.0 if int(params["weight_decay_is_zero"]) == 1 else float(params["weight_decay"]),
        "classifier_dropout": float(params["classifier_dropout"]),
        "entropy_lambda": 0.0 if int(params["entropy_lambda_is_zero"]) == 1 else float(params["entropy_lambda"]),
        "projector_dropout": float(params["projector_dropout"]),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    if y_true.size == 0:
        return {"accuracy": float("nan"), "benign_recall": float("nan"), "high_grade_recall": float("nan")}

    def _recall(label: int) -> float:
        mask = y_true == label
        if int(mask.sum()) == 0:
            return float("nan")
        return float(np.mean(y_pred[mask] == label))

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "benign_recall": _recall(0),
        "high_grade_recall": _recall(1),
    }


def metrics_table(fold_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """One row per outer fold, a pooled row, and the min/max across the five folds."""
    rows = []
    for row in fold_rows:
        rows.append(
            {
                "outer_test_fold": row["outer_test_fold"],
                "selected_configuration": row["selected_configuration"],
                "accuracy": row["accuracy"],
                "benign_recall": row["benign_recall"],
                "high_grade_recall": row["high_grade_recall"],
            }
        )
    pooled = classification_metrics(
        [y for row in fold_rows for y in row["y_true"]],
        [y for row in fold_rows for y in row["y_pred"]],
    )
    rows.append(
        {
            "outer_test_fold": "pooled",
            "selected_configuration": "differs by fold",
            "accuracy": pooled["accuracy"],
            "benign_recall": pooled["benign_recall"],
            "high_grade_recall": pooled["high_grade_recall"],
        }
    )
    frame = pd.DataFrame(rows)
    fold_only = frame[frame["outer_test_fold"].isin(GROUP_NAMES)]
    for stat_name, reducer in (("range_min", np.nanmin), ("range_max", np.nanmax)):
        rows.append(
            {
                "outer_test_fold": stat_name,
                "selected_configuration": "",
                "accuracy": float(reducer(fold_only["accuracy"].to_numpy(dtype=float))),
                "benign_recall": float(reducer(fold_only["benign_recall"].to_numpy(dtype=float))),
                "high_grade_recall": float(reducer(fold_only["high_grade_recall"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def load_groups(path: str) -> Dict[str, List[int]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing group file {path}. Run make_nested_splits.py first.")
    payload = np.load(path)
    groups = {}
    for name in GROUP_NAMES:
        key = f"group_{name}"
        if key not in payload:
            raise KeyError(f"{path} has no {key}")
        groups[name] = [int(value) for value in payload[key].tolist()]
    return groups


def inner_split_specs(groups: Dict[str, List[int]], outer: str) -> List[Dict[str, Any]]:
    """Four inner train/val splits. The outer test cases are never included."""
    if outer not in GROUP_NAMES:
        raise ValueError(f"Outer fold must be one of {GROUP_NAMES}, got {outer}")
    others = [name for name in GROUP_NAMES if name != outer]
    test_cases = set(groups[outer])
    specs = []
    for val_name in others:
        train_names = [name for name in others if name != val_name]
        train_cases = sorted(case for name in train_names for case in groups[name])
        val_cases = sorted(groups[val_name])
        overlap = set(train_cases) & set(val_cases) & test_cases
        if set(train_cases) & test_cases or set(val_cases) & test_cases or set(train_cases) & set(val_cases):
            raise RuntimeError(f"Inner split for outer {outer} / val {val_name} leaks cases: {sorted(overlap)}")
        specs.append(
            {
                "val_group": val_name,
                "train_groups": train_names,
                "train_cases": train_cases,
                "val_cases": val_cases,
            }
        )
    return specs


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_attention_entropy(attention_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Mean normalized patch-attention entropy. Attention itself is computed by the model."""
    eps = float(TRAINING_CONFIG.get("patch_entropy_eps", 1e-8))
    normalize = bool(TRAINING_CONFIG.get("normalize_patch_entropy", True))
    if not attention_info:
        return torch.zeros((), device=device)
    entropies = []
    for stain_info in attention_info.get("stain_weights", {}).values():
        for patch_weights in stain_info.get("patch_weights", []):
            if patch_weights is None or patch_weights.numel() == 0:
                continue
            probabilities = patch_weights.to(device).clamp_min(eps)
            entropy = -(probabilities * torch.log(probabilities)).sum()
            if normalize and probabilities.numel() > 1:
                entropy = entropy / torch.log(
                    torch.tensor(float(probabilities.numel()), device=device, dtype=entropy.dtype)
                )
            entropies.append(entropy)
    if not entropies:
        return torch.zeros((), device=device)
    return torch.stack(entropies).mean()


def _forward_case(model: nn.Module, case_data: Dict[str, Any], device: torch.device, need_entropy: bool):
    label = case_data["label"].to(device)
    if need_entropy:
        logits, attention = model(case_data["stain_slices"], return_attn_weights=True)
    else:
        logits = model(case_data["stain_slices"])
        attention = None
    if logits.dim() != 1:
        raise ValueError(f"Expected logits of shape (num_classes,), got {tuple(logits.shape)}")
    logits = logits.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)
    return logits, label, attention


class LockstepTrainer:
    """One inner model. Its scheduler watches its own balanced validation loss."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        class_weight_benign: float,
        class_weight_high: float,
        entropy_lambda: float,
    ):
        self.model = model.to(device)
        self.device = device
        self.entropy_lambda = float(entropy_lambda)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(TRAINING_CONFIG.get("scheduler_factor", 0.5)),
            patience=int(TRAINING_CONFIG.get("scheduler_patience", 4)),
            min_lr=float(TRAINING_CONFIG.get("scheduler_min_lr", 1e-6)),
        )
        weights = torch.tensor(
            [class_weight_benign, class_weight_high],
            dtype=torch.float32,
            device=device,
        )
        # Batch size is 1. reduction="mean" would divide by w_y and cancel the weight.
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        self.train_updates: List[float] = []
        self.train_epochs: List[float] = []
        self.val_epochs: List[float] = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            if not batch:
                continue
            need_entropy = self.entropy_lambda > 0.0
            logits, label, attention = _forward_case(self.model, batch[0], self.device, need_entropy)
            ce_loss = self.criterion(logits, label)
            if need_entropy:
                entropy = patch_attention_entropy(attention, self.device)
                loss = ce_loss - self.entropy_lambda * entropy
            else:
                loss = ce_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            value = float(loss.detach().item())
            losses.append(value)
            self.train_updates.append(value)
        if not losses:
            raise RuntimeError("Training loader produced no cases.")
        mean_loss = float(np.mean(losses))
        self.train_epochs.append(mean_loss)
        return mean_loss

    def validate(self, loader: DataLoader) -> float:
        """Balanced CE. Does not use training class weights or the entropy term."""
        loss = balanced_loader_loss(self.model, loader, self.device)
        self.val_epochs.append(loss)
        return loss


def run_lockstep(
    trainers: Sequence[LockstepTrainer],
    folds: Sequence[Dict[str, Any]],
    max_epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
) -> Dict[str, Any]:
    """
    Train the four models epoch by epoch.

    The epoch score is the average of the four balanced validation losses.
    selected_epochs is the epoch with the lowest average, not the average of
    four separately chosen epochs.
    """
    if len(trainers) != len(folds) or len(trainers) != 4:
        raise ValueError("Lockstep training expects four inner models.")
    best_mean = float("inf")
    selected_epochs = 0
    selected_fold_losses: List[float] = []
    stop_reference = float("inf")
    epochs_without_improvement = 0
    mean_curve: List[float] = []

    for epoch in range(max_epochs):
        fold_losses = []
        for trainer, fold in zip(trainers, folds):
            trainer.train_epoch(fold["train_loader"])
            val_loss = trainer.validate(fold["val_loader"])
            trainer.scheduler.step(val_loss)
            fold_losses.append(val_loss)
        mean_loss = float(np.mean(fold_losses))
        mean_curve.append(mean_loss)
        if mean_loss < best_mean:
            best_mean = mean_loss
            selected_epochs = epoch + 1
            selected_fold_losses = list(fold_losses)
        if mean_loss < stop_reference - min_delta:
            stop_reference = mean_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"  epoch {epoch + 1}/{max_epochs} "
            + " ".join(
                f"{fold['val_group']}={loss:.4f}" for fold, loss in zip(folds, fold_losses)
            )
            + f" mean={mean_loss:.4f}"
        )
        if (epoch + 1) >= min_epochs and epochs_without_improvement >= patience:
            print(f"  early stop at epoch {epoch + 1}; best mean epoch is {selected_epochs}")
            break

    chosen_epoch, chosen_value = best_epoch_from_curve(mean_curve)
    if chosen_epoch != selected_epochs or not math.isclose(chosen_value, best_mean):
        raise RuntimeError("Selected epoch does not match the minimum of the mean curve.")
    return {
        "selected_epochs": selected_epochs,
        "best_mean": best_mean,
        "fold_losses": selected_fold_losses,
        "mean_curve": mean_curve,
    }


def build_model(hparams: Dict[str, float], seed: int, pooled_dim: int) -> HierarchicalAttnMIL:
    seed_all(seed)
    return HierarchicalAttnMIL(
        num_classes=MODEL_CONFIG["num_classes"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        patch_proj_dropout=hparams["projector_dropout"],
        classifier_dropout=hparams["classifier_dropout"],
        pooled_dim=pooled_dim,
    )


def prepare_case_index(labels_csv: str, patches_dir: str):
    labels = load_labels(labels_csv)
    all_files = get_all_patch_files(patches_dir)
    patches = group_patches_by_slice(all_files, patches_dir)
    slice_to_class = build_slice_to_class_map(patches, labels)
    return patches, slice_to_class


def _slices_for_cases(slice_to_class: dict, case_ids: Sequence[int]):
    wanted = set(int(case) for case in case_ids)
    return [key for key in slice_to_class if int(key[0]) in wanted]


def make_dataset(
    case_ids: Sequence[int],
    patches,
    slice_to_class,
    embeddings_dir: str,
    per_slice_cap: int,
    max_slices_per_stain: Optional[int],
    shuffle_patches: bool,
) -> Tuple[StainBagCasePooledFeatureDataset, Dict[int, int]]:
    slices = _slices_for_cases(slice_to_class, case_ids)
    case_dict, label_map = build_case_dict(slices, patches, slice_to_class)
    missing = sorted(set(int(c) for c in case_ids) - set(int(c) for c in label_map))
    if missing:
        raise RuntimeError(f"Cases in the split have no usable patches: {missing[:12]}")
    dataset = StainBagCasePooledFeatureDataset(
        case_dict,
        label_map,
        embeddings_dir=embeddings_dir,
        per_slice_cap=per_slice_cap,
        max_slices_per_stain=max_slices_per_stain,
        shuffle_patches=shuffle_patches,
    )
    return dataset, {int(k): int(v) for k, v in label_map.items()}


def make_loader(dataset, shuffle: bool, seed: int, num_workers: int, pin_memory: bool) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=case_collate_fn,
        generator=generator,
    )


def infer_pooled_dim(embeddings_dir: str) -> int:
    with os.scandir(embeddings_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".pt"):
                tensor = torch.load(entry.path, map_location="cpu")
                return int(tensor.view(-1).numel())
    raise FileNotFoundError(f"No .pt embeddings found in {embeddings_dir}")


def _save_trial_curves(path: str, trainers: Sequence[LockstepTrainer], folds, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "selected_epochs": np.array([result["selected_epochs"]], dtype=np.int64),
        "mean_curve": np.asarray(result["mean_curve"], dtype=np.float32),
    }
    for trainer, fold in zip(trainers, folds):
        name = fold["val_group"]
        payload[f"train_updates_{name}"] = np.asarray(trainer.train_updates, dtype=np.float32)
        payload[f"train_epochs_{name}"] = np.asarray(trainer.train_epochs, dtype=np.float32)
        payload[f"val_epochs_{name}"] = np.asarray(trainer.val_epochs, dtype=np.float32)
    np.savez_compressed(path, **payload)


def attach_loaders(
    folds: Sequence[Dict[str, Any]],
    seed: int,
    outer_index: int,
    num_workers: int,
    pin_memory: bool,
) -> None:
    """New loaders each trial so the shuffle seed does not drift across trials."""
    for inner_index, fold in enumerate(folds):
        fold["train_loader"] = make_loader(
            fold["train_dataset"],
            True,
            seed + outer_index * 10 + inner_index,
            num_workers,
            pin_memory,
        )
        fold["val_loader"] = make_loader(
            fold["val_dataset"],
            False,
            seed,
            num_workers,
            pin_memory,
        )


def run_one_trial(
    trial: optuna.Trial,
    folds: Sequence[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    pooled_dim: int,
    base_seed: int,
    outer_index: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    curve_path: str,
    num_workers: int,
) -> float:
    hparams = suggest_hparams(trial, bounds)
    print("\n" + "=" * 80)
    print(f"Trial {trial.number}")
    for key in PARAM_ORDER:
        value = hparams[key]
        print(f"  {key}: {value:.6g}")

    attach_loaders(folds, base_seed, outer_index, num_workers, device.type == "cuda")
    trainers = []
    try:
        for inner_index, _fold in enumerate(folds):
            model = build_model(
                hparams,
                seed=base_seed + outer_index * 10 + inner_index,
                pooled_dim=pooled_dim,
            )
            weight_b, weight_h, _n_b, _n_h = folds[inner_index]["class_weights"]
            trainers.append(
                LockstepTrainer(
                    model=model,
                    device=device,
                    learning_rate=hparams["learning_rate"],
                    weight_decay=hparams["weight_decay"],
                    class_weight_benign=weight_b,
                    class_weight_high=weight_h,
                    entropy_lambda=hparams["entropy_lambda"],
                )
            )
        result = run_lockstep(
            trainers,
            folds,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            min_epochs=min_epochs,
        )
        _save_trial_curves(curve_path, trainers, folds, result)
        trial.set_user_attr("selected_epochs", int(result["selected_epochs"]))
        trial.set_user_attr("mean_val_at_selected", float(result["best_mean"]))
        trial.set_user_attr("mean_val_curve", [float(v) for v in result["mean_curve"]])
        for fold, loss in zip(folds, result["fold_losses"]):
            trial.set_user_attr(f"val_{fold['val_group']}_at_selected", float(loss))
        return float(result["best_mean"])
    finally:
        del trainers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _study_storage(directory: str) -> str:
    return "sqlite:///" + os.path.join(directory, "optuna.db")


def run_study(
    directory: str,
    study_name: str,
    bounds: Dict[str, Tuple[float, float]],
    folds: Sequence[Dict[str, Any]],
    device: torch.device,
    pooled_dim: int,
    base_seed: int,
    outer_index: int,
    n_trials: int,
    sobol_points: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    num_workers: int,
) -> optuna.Study:
    os.makedirs(directory, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=base_seed + outer_index,
        n_startup_trials=0,
        gamma=good_set_size,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=_study_storage(directory),
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        for point in sobol_startup_points(sobol_points, base_seed, bounds):
            study.enqueue_trial(point)
        print(f"Enqueued {sobol_points} scrambled Sobol trials in {directory}")
    # Waiting trials are the Sobol queue. They are consumed inside optimize,
    # so only trials that already left the queue count against the budget.
    consumed = sum(trial.state != optuna.trial.TrialState.WAITING for trial in study.trials)
    remaining = n_trials - consumed
    if remaining > 0:
        def objective(trial: optuna.Trial) -> float:
            curve_path = os.path.join(directory, "curves", f"trial_{trial.number:04d}.npz")
            return run_one_trial(
                trial,
                folds,
                bounds,
                device,
                pooled_dim,
                base_seed,
                outer_index,
                max_epochs,
                patience,
                min_delta,
                min_epochs,
                curve_path,
                num_workers=num_workers,
            )

        study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"Study {study_name} has no completed trials.")
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    frame.to_csv(os.path.join(directory, "trials.csv"), index=False)
    return study


def refit_model(
    hparams: Dict[str, float],
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: Tuple[float, float, int, int],
    device: torch.device,
    pooled_dim: int,
    seed: int,
) -> Tuple[HierarchicalAttnMIL, List[float], List[float], List[float]]:
    """
    Train on the four non-test groups for exactly `epochs`.

    Hyperparameters and `epochs` are already fixed by the inner search.
    After each epoch the held-out loss is recorded with no gradient. It is
    not passed to the scheduler, and it cannot stop training or choose a
    checkpoint. The returned model is the one that finished epoch `epochs`.
    """
    if epochs < 1:
        raise ValueError(f"Refit epoch count must be positive, got {epochs}")
    model = build_model(hparams, seed, pooled_dim)
    weight_b, weight_h, _n_b, _n_h = class_weights
    trainer = LockstepTrainer(
        model=model,
        device=device,
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        class_weight_benign=weight_b,
        class_weight_high=weight_h,
        entropy_lambda=hparams["entropy_lambda"],
    )
    # The trainer builds a scheduler for the inner loop. Refit never steps it,
    # so the learning rate stays at the value chosen by the inner search.
    test_epochs: List[float] = []
    for epoch in range(epochs):
        mean_loss = trainer.train_epoch(train_loader)
        test_loss = balanced_loader_loss(trainer.model, test_loader, device)
        test_epochs.append(test_loss)
        print(
            f"  refit epoch {epoch + 1}/{epochs} "
            f"train_loss={mean_loss:.4f} test_loss={test_loss:.4f}"
        )
    if len(trainer.train_epochs) != epochs or len(test_epochs) != epochs:
        raise RuntimeError(
            f"Refit stopped before epoch {epochs}: "
            f"train={len(trainer.train_epochs)}, test={len(test_epochs)}"
        )
    return model, trainer.train_updates, trainer.train_epochs, test_epochs


def evaluate_once(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Predictions on the model that finished the predetermined refit epoch."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            if not batch:
                continue
            case = batch[0]
            logits, label, _attention = _forward_case(model, case, device, False)
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = int(torch.argmax(logits, dim=1).item())
            true_label = int(label.item())
            rows.append(
                {
                    "case_id": int(case["case_id"]),
                    "true_label": true_label,
                    "predicted_label": prediction,
                    "prob_benign": float(probabilities[0].item()),
                    "prob_high_grade": float(probabilities[1].item()),
                    "correct": prediction == true_label,
                }
            )
    if not rows:
        raise RuntimeError("The held-out evaluation saw no cases.")
    return pd.DataFrame(rows)


def _plot_inner_val_loss(
    path: str,
    val_epochs: Dict[str, np.ndarray],
    mean_curve: Sequence[float],
    selected: int,
    outer_name: str,
) -> None:
    """Four inner validation curves plus their mean. Epoch ticks are integers."""
    figure, axis = plt.subplots(figsize=(8, 5))
    length = 0
    for name, values in val_epochs.items():
        values = np.asarray(values, dtype=np.float32)
        length = max(length, len(values))
        axis.plot(np.arange(1, len(values) + 1), values, label=name)
    mean = np.asarray(mean_curve, dtype=np.float32)
    length = max(length, len(mean))
    axis.plot(
        np.arange(1, len(mean) + 1),
        mean,
        color="black",
        linewidth=2.0,
        label="mean validation loss",
    )
    axis.axvline(selected, color="0.4", linestyle="--", linewidth=1, label=f"selected epoch {selected}")
    axis.set_xlim(0.5, length + 0.5)
    axis.xaxis.set_major_locator(MultipleLocator(1))
    axis.set_xlabel("epoch")
    axis.set_ylabel("balanced validation loss")
    axis.set_title(f"Selected trial: inner validation loss per epoch\nOuter {outer_name}")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_lines(path: str, series: Dict[str, np.ndarray], xlabel: str, ylabel: str, title: str, vline: Optional[int] = None) -> None:
    plt.figure(figsize=(8, 5))
    for name, values in series.items():
        plt.plot(np.arange(1, len(values) + 1), values, label=name)
    if vline is not None:
        plt.axvline(vline, color="black", linestyle="--", linewidth=1, label=f"selected epoch {vline}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_refit_train_test(
    path: str,
    train_epochs: Sequence[float],
    test_epochs: Sequence[float],
    outer_name: str,
) -> None:
    """Train and test loss on one shared y-axis. The line marks the reported epoch."""
    train = np.asarray(train_epochs, dtype=np.float32)
    test = np.asarray(test_epochs, dtype=np.float32)
    if len(train) == 0 or len(train) != len(test):
        raise RuntimeError("Refit train and test curves must have the same positive length.")
    reported = int(len(train))
    epochs = np.arange(1, reported + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train, color="tab:blue", label="training loss")
    axis.plot(epochs, test, color="tab:orange", label="balanced test loss")
    axis.axvline(reported, color="black", linestyle="--", linewidth=1, label=f"reported epoch {reported}")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend(loc="best")
    axis.set_title(f"Refit training loss and held-out test loss\nOuter {outer_name}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_curve_plots(
    curve_file: str,
    refit_updates: Sequence[float],
    refit_epochs: Sequence[float],
    refit_test_epochs: Sequence[float],
    plot_dir: str,
    outer_name: str,
) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    stored = np.load(curve_file)
    selected = int(stored["selected_epochs"][0])
    train_updates = {}
    train_epochs = {}
    val_epochs = {}
    for name in GROUP_NAMES:
        update_key = f"train_updates_{name}"
        if update_key not in stored:
            continue
        train_updates[f"val {name}"] = stored[update_key]
        train_epochs[f"val {name}"] = stored[f"train_epochs_{name}"]
        val_epochs[f"val {name}"] = stored[f"val_epochs_{name}"]
    _plot_lines(
        os.path.join(plot_dir, "inner_train_loss_per_update.png"),
        train_updates,
        "update",
        "training loss",
        "Selected trial: inner training loss per update",
    )
    _plot_lines(
        os.path.join(plot_dir, "inner_train_loss_per_epoch.png"),
        train_epochs,
        "epoch",
        "training loss",
        "Selected trial: inner training loss per epoch",
        vline=selected,
    )
    _plot_inner_val_loss(
        os.path.join(plot_dir, "inner_val_loss_per_epoch.png"),
        val_epochs,
        stored["mean_curve"],
        selected,
        outer_name,
    )
    _plot_lines(
        os.path.join(plot_dir, "refit_train_loss_per_update.png"),
        {"refit": np.asarray(refit_updates, dtype=np.float32)},
        "update",
        "training loss",
        "Refit training loss per update",
    )
    _plot_refit_train_test(
        os.path.join(plot_dir, "refit_train_test_loss_per_epoch.png"),
        refit_epochs,
        refit_test_epochs,
        outer_name,
    )


def save_probability_plot(predictions: pd.DataFrame, path: str) -> None:
    rng = np.random.default_rng(0)
    x = predictions["true_label"].to_numpy(dtype=float)
    x = x + rng.uniform(-0.08, 0.08, size=len(x))
    plt.figure(figsize=(6, 5))
    plt.scatter(x, predictions["prob_high_grade"], s=28, alpha=0.85)
    plt.xticks([0, 1], ["benign", "high-grade"])
    plt.ylabel("predicted probability of high-grade")
    plt.xlabel("true label")
    plt.ylim(-0.05, 1.05)
    plt.title("Held-out predicted probability vs true label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def configuration_text(hparams: Dict[str, float], selected_epochs: int) -> str:
    ordered = {key: hparams[key] for key in PARAM_ORDER}
    ordered["selected_epochs"] = selected_epochs
    return json.dumps(ordered, sort_keys=False)


def write_pooled_results(output_dir: str) -> Optional[str]:
    fold_rows = []
    for name in GROUP_NAMES:
        pred_path = os.path.join(output_dir, f"outer_{name}", "predictions.csv")
        meta_path = os.path.join(output_dir, f"outer_{name}", "best_params.json")
        if not (os.path.isfile(pred_path) and os.path.isfile(meta_path)):
            return None
        predictions = pd.read_csv(pred_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        metrics = classification_metrics(predictions["true_label"], predictions["predicted_label"])
        fold_rows.append(
            {
                "outer_test_fold": name,
                "selected_configuration": configuration_text(meta["hparams"], int(meta["selected_epochs"])),
                "y_true": predictions["true_label"].tolist(),
                "y_pred": predictions["predicted_label"].tolist(),
                **metrics,
            }
        )
    table = metrics_table(fold_rows)
    path = os.path.join(output_dir, "nested_results.csv")
    table.to_csv(path, index=False)
    print(f"Wrote {path}")
    return path


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def run_outer_fold(args, outer: str, prepared: Dict[str, Any], device: torch.device) -> None:
    outer_index = GROUP_NAMES.index(outer)
    outer_dir = os.path.join(args.output_dir, f"outer_{outer}")
    os.makedirs(outer_dir, exist_ok=True)
    test_cases = set(prepared["groups"][outer])
    folds = []
    for spec in inner_split_specs(prepared["groups"], outer):
        if set(spec["train_cases"]) & test_cases or set(spec["val_cases"]) & test_cases:
            raise RuntimeError(f"Outer group {outer} leaked into an inner split.")
        train_dataset, train_labels = make_dataset(
            spec["train_cases"],
            prepared["patches"],
            prepared["slice_to_class"],
            args.embeddings_dir,
            args.per_slice_cap,
            args.max_slices_per_stain,
            shuffle_patches=True,
        )
        val_dataset, _val_labels = make_dataset(
            spec["val_cases"],
            prepared["patches"],
            prepared["slice_to_class"],
            args.embeddings_dir,
            args.per_slice_cap,
            args.max_slices_per_stain,
            shuffle_patches=False,
        )
        weights = class_weights_from_labels(train_labels.values())
        folds.append(
            {
                **spec,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "class_weights": weights,
            }
        )
        print(
            f"Outer {outer} | train {''.join(spec['train_groups'])} n={len(train_dataset)} "
            f"weights benign={weights[0]:.4f} high={weights[1]:.4f} | val {spec['val_group']} n={len(val_dataset)}"
        )

    initial_bounds = {key: tuple(value) for key, value in DEFAULT_BOUNDS.items()}
    study = run_study(
        directory=os.path.join(outer_dir, "study"),
        study_name=f"nested_outer_{outer}",
        bounds=initial_bounds,
        folds=folds,
        device=device,
        pooled_dim=prepared["pooled_dim"],
        base_seed=args.seed,
        outer_index=outer_index,
        n_trials=args.n_trials,
        sobol_points=args.sobol_startup_trials,
        max_epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
        num_workers=args.num_workers,
    )
    first_decoded = decode_hparams(study.best_trial.params)
    assessment = assess_edges(first_decoded, initial_bounds)
    print(f"Outer {outer} boundary check:")
    for note in assessment["notes"]:
        print(f"  {note}")

    chosen_study = study
    bounds_used = initial_bounds
    widened = False
    if assessment["changed"]:
        widened = True
        print(f"Outer {outer}: rerunning {args.n_trials} trials with widened bounds.")
        chosen_study = run_study(
            directory=os.path.join(outer_dir, "study_widened"),
            study_name=f"nested_outer_{outer}_widened",
            bounds=assessment["bounds"],
            folds=folds,
            device=device,
            pooled_dim=prepared["pooled_dim"],
            base_seed=args.seed + 1000,
            outer_index=outer_index,
            n_trials=args.n_trials,
            sobol_points=args.sobol_startup_trials,
            max_epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            num_workers=args.num_workers,
        )
        bounds_used = assessment["bounds"]

    best = chosen_study.best_trial
    hparams = decode_hparams(best.params)
    selected_epochs = int(best.user_attrs["selected_epochs"])
    if selected_epochs < 1:
        raise RuntimeError(f"Selected epoch must be positive, got {selected_epochs}")
    final_edge = assess_edges(hparams, bounds_used)

    refit_cases = sorted(case for name, cases in prepared["groups"].items() if name != outer for case in cases)
    if set(refit_cases) & test_cases:
        raise RuntimeError("Refit cases overlap the held-out group.")
    refit_dataset, refit_labels = make_dataset(
        refit_cases,
        prepared["patches"],
        prepared["slice_to_class"],
        args.embeddings_dir,
        args.per_slice_cap,
        args.max_slices_per_stain,
        shuffle_patches=True,
    )
    refit_weights = class_weights_from_labels(refit_labels.values())
    refit_loader = make_loader(
        refit_dataset,
        True,
        args.seed + outer_index,
        args.num_workers,
        device.type == "cuda",
    )
    test_dataset, _test_labels = make_dataset(
        sorted(test_cases),
        prepared["patches"],
        prepared["slice_to_class"],
        args.embeddings_dir,
        args.per_slice_cap,
        args.max_slices_per_stain,
        shuffle_patches=False,
    )
    test_loader = make_loader(test_dataset, False, args.seed, args.num_workers, device.type == "cuda")
    print(
        f"Refit outer {outer} on {len(refit_dataset)} cases for {selected_epochs} epochs. "
        f"Test loss is recorded on {len(test_dataset)} held-out cases and is not used to "
        f"change the learning rate, the epoch count, or the reported metrics."
    )
    model, refit_updates, refit_epochs, refit_test_epochs = refit_model(
        hparams,
        selected_epochs,
        refit_loader,
        test_loader,
        refit_weights,
        device,
        prepared["pooled_dim"],
        seed=args.seed + outer_index,
    )
    checkpoint_path = os.path.join(outer_dir, "refit.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hparams": hparams,
            "selected_epochs": selected_epochs,
            "pooled_dim": prepared["pooled_dim"],
            "refit_train_loss_per_epoch": refit_epochs,
            "refit_test_loss_per_epoch": refit_test_epochs,
        },
        checkpoint_path,
    )
    curve_path = os.path.join(outer_dir, "refit_curves.npz")
    np.savez_compressed(
        curve_path,
        train_updates=np.asarray(refit_updates, dtype=np.float32),
        train_epochs=np.asarray(refit_epochs, dtype=np.float32),
        test_epochs=np.asarray(refit_test_epochs, dtype=np.float32),
        reported_epoch=np.array([selected_epochs], dtype=np.int64),
    )

    predictions = evaluate_once(model, test_loader, device)
    pred_path = os.path.join(outer_dir, "predictions.csv")
    predictions.to_csv(pred_path, index=False)
    metrics = classification_metrics(predictions["true_label"], predictions["predicted_label"])
    print(
        f"Outer {outer} reported epoch {selected_epochs}: accuracy={metrics['accuracy']:.4f} "
        f"benign_recall={metrics['benign_recall']:.4f} "
        f"high_grade_recall={metrics['high_grade_recall']:.4f}"
    )

    study_dir = os.path.join(outer_dir, "study_widened" if widened else "study")
    curve_file = os.path.join(study_dir, "curves", f"trial_{best.number:04d}.npz")
    plot_dir = os.path.join(outer_dir, "plots")
    if os.path.isfile(curve_file):
        save_curve_plots(curve_file, refit_updates, refit_epochs, refit_test_epochs, plot_dir, outer)
    else:
        print(f"Missing inner curve file {curve_file}; refit plots will still be written.")
        save_curve_plots_refit_only(refit_updates, refit_epochs, refit_test_epochs, plot_dir, outer)
    save_probability_plot(predictions, os.path.join(plot_dir, "pred_prob_vs_label.png"))

    record = {
        "outer_test_fold": outer,
        "hparams": hparams,
        "selected_epochs": selected_epochs,
        "mean_val_at_selected": float(best.user_attrs.get("mean_val_at_selected", best.value)),
        "val_at_selected": {
            name: best.user_attrs.get(f"val_{name}_at_selected")
            for name in GROUP_NAMES
            if name != outer
        },
        "refit_class_weights": {
            "benign": refit_weights[0],
            "high_grade": refit_weights[1],
            "n_benign": refit_weights[2],
            "n_high": refit_weights[3],
        },
        "widened": widened,
        "bounds_used": {key: list(value) for key, value in bounds_used.items()},
        "boundary_notes": assessment["notes"],
        "still_on_expandable_edge": bool(final_edge["changed"]),
        "final_edge_notes": final_edge["notes"],
        "reported_epoch": selected_epochs,
        "refit_train_loss_per_epoch": [float(value) for value in refit_epochs],
        "refit_test_loss_per_epoch": [float(value) for value in refit_test_epochs],
        "metrics": metrics,
        "checkpoint": checkpoint_path,
        "predictions": pred_path,
        "refit_curves": curve_path,
    }
    with open(os.path.join(outer_dir, "best_params.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_ready(record), handle, indent=2)
    write_pooled_results(args.output_dir)


def save_curve_plots_refit_only(refit_updates, refit_epochs, refit_test_epochs, plot_dir: str, outer_name: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    _plot_lines(
        os.path.join(plot_dir, "refit_train_loss_per_update.png"),
        {"refit": np.asarray(refit_updates, dtype=np.float32)},
        "update",
        "training loss",
        "Refit training loss per update",
    )
    _plot_refit_train_test(
        os.path.join(plot_dir, "refit_train_test_loss_per_epoch.png"),
        refit_epochs,
        refit_test_epochs,
        outer_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested cross-validation with one Optuna study per outer fold.")
    parser.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
    parser.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument(
        "--groups_npz",
        type=str,
        default=os.path.join(SCRIPT_DIR, "nested_splits", "nested_groups.npz"),
    )
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "nested_cv_runs"))
    parser.add_argument("--outer_fold", type=str, default="all", choices=["all", *GROUP_NAMES])
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--sobol_startup_trials", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=int(TRAINING_CONFIG["epochs"]))
    parser.add_argument("--patience", type=int, default=int(TRAINING_CONFIG["early_stopping_patience"]))
    parser.add_argument("--min_delta", type=float, default=float(TRAINING_CONFIG["early_stopping_min_delta"]))
    parser.add_argument("--min_epochs", type=int, default=int(TRAINING_CONFIG["early_stopping_min_epochs"]))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=int(TRAINING_CONFIG["num_workers"]))
    parser.add_argument("--per_slice_cap", type=int, default=500)
    parser.add_argument("--max_slices_per_stain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=int(TRAINING_CONFIG["random_state"]))
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("Batch size must be 1 so weighted CE with reduction='sum' keeps the class weight.")
    if args.sobol_startup_trials > args.n_trials:
        raise ValueError("Sobol startup count cannot exceed n_trials.")
    if args.max_slices_per_stain < 0:
        args.max_slices_per_stain = None
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(get_device())
    groups = load_groups(args.groups_npz)
    patches, slice_to_class = prepare_case_index(args.labels_csv, args.patches_dir)
    pooled_dim = infer_pooled_dim(args.embeddings_dir)
    print(f"Inferred pooled_dim={pooled_dim} from {args.embeddings_dir}")
    prepared = {
        "groups": groups,
        "patches": patches,
        "slice_to_class": slice_to_class,
        "pooled_dim": pooled_dim,
    }
    outers = list(GROUP_NAMES) if args.outer_fold == "all" else [args.outer_fold]
    for outer in outers:
        print("=" * 80)
        print(f"OUTER TEST GROUP {outer}")
        print("=" * 80)
        run_outer_fold(args, outer, prepared, device)
    write_pooled_results(args.output_dir)


if __name__ == "__main__":
    main()
