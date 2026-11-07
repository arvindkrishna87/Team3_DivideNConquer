"""
Dataset classes for MIL training
"""
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG


class StainBagCasePooledFeatureDataset(Dataset):
    """
    Like StainBagCaseDataset, but loads precomputed pooled feature vectors (.pt)
    instead of loading images.

    For each CASE returns:
    {
        "case_id": int/str,
        "stain_slices": {
            "h&e":   [Tensor(P1, F4), Tensor(P2, F4), ...],
            ...
        },
        "label": LongTensor scalar
    }
    where F4 = 4*F (typically 4096 for DenseNet121 with AdaptiveAvgPool2d((2,2))).
    """

    def __init__(
        self,
        case_dict: Dict[Any, Dict[str, List[List[str]]]],
        label_map: Dict[Any, int],
        embeddings_dir: str,
        stains: Tuple[str, ...] = None,
        per_slice_cap: Optional[int] = None,
        max_slices_per_stain: Optional[int] = None,
        shuffle_patches: bool = True,
        drop_empty_slices: bool = True,
        # NEW: force a consistent dtype so fp16 embeddings won't crash fp32 model
        feature_dtype: torch.dtype = torch.float32,
    ):
        self.embeddings_dir = embeddings_dir
        self.stains = list(stains) if stains else list(MODEL_CONFIG["stains"])
        self.per_slice_cap = per_slice_cap if per_slice_cap is not None else MODEL_CONFIG["per_slice_cap"]
        self.max_slices_per_stain = (
            max_slices_per_stain if max_slices_per_stain is not None else MODEL_CONFIG["max_slices_per_stain"]
        )
        self.shuffle_patches = shuffle_patches
        self.drop_empty_slices = drop_empty_slices

        # NEW
        self.feature_dtype = feature_dtype
        # Full slice tensors, loaded once. Later accesses only shuffle and cap.
        self._slice_cache: Dict[Tuple[str, ...], Optional[torch.Tensor]] = {}

        self.items = []
        for case_id, stain_map in case_dict.items():
            if case_id not in label_map:
                continue
            norm_map = {k.lower(): v for k, v in stain_map.items()}
            self.items.append((case_id, norm_map))
        self.label_map = label_map

    def __len__(self):
        return len(self.items)

    def _patch_path_to_feat_path(self, patch_path: str) -> str:
        # patch_path is full path like /projects/.../patches/<fname>.png
        fname = os.path.basename(patch_path)
        feat_name = os.path.splitext(fname)[0] + ".pt"
        return os.path.join(self.embeddings_dir, feat_name)

    def _cache_slice(self, patch_paths: List[str]) -> Optional[torch.Tensor]:
        """Load every patch in this slice once. Missing files are skipped."""
        cache_key = tuple(patch_paths)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        vecs = []
        for p in patch_paths:
            fp = self._patch_path_to_feat_path(p)
            try:
                v = torch.load(fp, map_location="cpu")  # (4096,) or (1536,)
                if v is None:
                    continue

                if v.dim() != 1:
                    v = v.view(-1)

                # Half embeddings must be cast before they reach the float32 model.
                if v.dtype != self.feature_dtype:
                    v = v.to(self.feature_dtype)

                vecs.append(v)
            except Exception:
                continue

        full = torch.stack(vecs, dim=0) if vecs else None
        self._slice_cache[cache_key] = full
        return full

    def preload(self) -> None:
        """Read every slice used by this dataset into memory before training."""
        n_cases = len(self.items)
        for case_i, (_, stain_map) in enumerate(self.items, start=1):
            for stain in self.stains:
                slice_lists = stain_map.get(stain, []) or []
                if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                    slice_lists = slice_lists[: self.max_slices_per_stain]
                for sl in slice_lists:
                    if sl:
                        self._cache_slice(sl)
            if case_i % 10 == 0 or case_i == n_cases:
                print(f"Cached embeddings for {case_i}/{n_cases} cases", flush=True)

    def _load_slice_tensor(self, patch_paths: List[str]) -> Optional[torch.Tensor]:
        full = self._cache_slice(patch_paths)
        if full is None:
            return None

        n = full.size(0)
        if self.shuffle_patches:
            perm = torch.randperm(n)
        else:
            perm = torch.arange(n)
        if self.per_slice_cap and n > self.per_slice_cap:
            perm = perm[: self.per_slice_cap]
        return full.index_select(0, perm)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, stain_map = self.items[idx]
        label = torch.tensor(self.label_map[case_id], dtype=torch.long)

        stain_slices: Dict[str, List[torch.Tensor]] = {}
        for stain in self.stains:
            slice_lists = stain_map.get(stain, []) or []

            if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                slice_lists = slice_lists[:self.max_slices_per_stain]

            tensors_for_stain: List[torch.Tensor] = []
            for sl in slice_lists:
                if not sl:
                    continue
                sl_tensor = self._load_slice_tensor(sl)

                if sl_tensor is None:
                    if self.drop_empty_slices:
                        # skip empty slice instead of crashing
                        continue
                    raise RuntimeError(f"Empty slice encountered in case {case_id}, stain {stain}")

                tensors_for_stain.append(sl_tensor)

            stain_slices[stain] = tensors_for_stain

        return {"case_id": case_id, "stain_slices": stain_slices, "label": label}


def case_collate_fn(batch):
    """
    Collate function for MIL dataset
    Keep variable-length structures; model.forward expects List[case_dict]
    Use batch_size=1 to avoid padding/masking
    """
    return batch