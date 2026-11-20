"""
Dataset classes for MIL training

Presentation 2, Analyst choice 1 changes:
  * Every patch tensor travels with its own filename. __getitem__ returns
    "stain_paths", aligned row-for-row with "stain_slices", so attention scores
    can be matched to images by name instead of by sort order.
  * No silent failures: a missing/unreadable embedding, an empty slice, or a
    malformed tensor raises an error that names the file, instead of being skipped.
"""
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG

# One process-wide copy, keyed by directory and dtype. Later datasets reuse it.
_EMBEDDING_CACHE: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
_WARNED_MISSING: set = set()


def get_embedding_cache(embeddings_dir: str, feature_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Load every .pt in embeddings_dir once. The returned dict is owned by this process."""
    key = (os.path.abspath(embeddings_dir), str(feature_dtype))
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached

    names = sorted(name for name in os.listdir(embeddings_dir) if name.endswith(".pt"))
    print(f"Loading {len(names)} embeddings from {key[0]} into memory...")
    store: Dict[str, torch.Tensor] = {}
    for index, name in enumerate(names, start=1):
        path = os.path.join(key[0], name)
        try:
            value = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Could not load embedding {path}") from exc
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Embedding {path} is {type(value).__name__}, expected a tensor")
        if value.dim() != 1:
            value = value.view(-1)
        if value.dtype != feature_dtype:
            value = value.to(feature_dtype)
        store[name] = value
        if index % 10000 == 0 or index == len(names):
            print(f"  loaded {index}/{len(names)}")
    _EMBEDDING_CACHE[key] = store
    print(f"Embedding cache ready: {len(store)} vectors")
    return store


class StainBagCasePooledFeatureDataset(Dataset):
    """
    Loads precomputed pooled feature vectors (.pt) for each patch.

    For each CASE returns:
    {
        "case_id": int/str,
        "stain_slices": {
            "h&e": [Tensor(P1, F4), Tensor(P2, F4), ...],
            ...
        },
        "stain_paths": {                     # NEW: same nesting as stain_slices
            "h&e": [[png_path_row0, png_path_row1, ...], ...],
            ...
        },
        "label": LongTensor scalar
    }
    stain_paths[stain][s][i] is the patch image behind stain_slices[stain][s][i].
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
        feature_dtype: torch.dtype = torch.float32,
    ):
        # drop_empty_slices was removed: empty slices now raise (see __getitem__).
        self.embeddings_dir = embeddings_dir
        self.stains = list(stains) if stains else list(MODEL_CONFIG["stains"])
        self.per_slice_cap = per_slice_cap if per_slice_cap is not None else MODEL_CONFIG["per_slice_cap"]
        self.max_slices_per_stain = (
            max_slices_per_stain if max_slices_per_stain is not None else MODEL_CONFIG["max_slices_per_stain"]
        )
        self.shuffle_patches = shuffle_patches
        self.feature_dtype = feature_dtype
        self._embeddings = get_embedding_cache(embeddings_dir, self.feature_dtype)

        self.items = []
        for case_id, stain_map in case_dict.items():
            if case_id not in label_map:
                raise KeyError(f"Case {case_id} has patches but no label in label_map")
            norm_map = {k.lower(): v for k, v in stain_map.items()}
            self.items.append((case_id, norm_map))
        self.label_map = label_map

    def __len__(self):
        return len(self.items)

    def _feat_name(self, patch_path: str) -> str:
        return os.path.splitext(os.path.basename(patch_path))[0] + ".pt"

    def _load_patch_vector(self, patch_path: str) -> torch.Tensor:
        feat_name = self._feat_name(patch_path)
        value = self._embeddings.get(feat_name)
        if value is None:
            raise FileNotFoundError(
                f"Embedding not found for patch {patch_path}: {feat_name}"
            )
        return value

    def _load_slice_tensor(self, patch_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """Return (tensor of shape (P, F), the P patch paths in the same row order)."""
        patch_paths = list(patch_paths)

        # Patches with no embedding (smaller than 32x32, or skipped at encode time)
        # are left out. Each missing file is printed once, not on every epoch.
        missing = [p for p in patch_paths if self._feat_name(p) not in self._embeddings]
        if missing:
            fresh = [p for p in missing if p not in _WARNED_MISSING]
            _WARNED_MISSING.update(missing)
            if fresh:
                print(
                    f"EXCLUDED {len(fresh)} patch(es) with no embedding: "
                    + ", ".join(os.path.basename(p) for p in fresh)
                )
            patch_paths = [p for p in patch_paths if p not in missing]
        if not patch_paths:
            raise ValueError("Every patch in this slice is missing its embedding")

        if self.shuffle_patches:
            random.shuffle(patch_paths)  # paths are shuffled, so names follow their rows
        if self.per_slice_cap and len(patch_paths) > self.per_slice_cap:
            patch_paths = patch_paths[: self.per_slice_cap]

        vecs = [self._load_patch_vector(p) for p in patch_paths]

        sizes = {v.numel() for v in vecs}
        if len(sizes) != 1:
            raise ValueError(f"Embeddings in one slice have different lengths {sorted(sizes)}: "
                             f"{os.path.basename(patch_paths[0])} ...")
        return torch.stack(vecs, dim=0), patch_paths

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, stain_map = self.items[idx]
        label = torch.tensor(self.label_map[case_id], dtype=torch.long)

        stain_slices: Dict[str, List[torch.Tensor]] = {}
        stain_paths: Dict[str, List[List[str]]] = {}
        for stain in self.stains:
            slice_lists = stain_map.get(stain, []) or []

            if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                slice_lists = slice_lists[: self.max_slices_per_stain]

            tensors_for_stain: List[torch.Tensor] = []
            paths_for_stain: List[List[str]] = []
            for s_i, sl in enumerate(slice_lists):
                if not sl:
                    raise ValueError(f"Case {case_id}, stain {stain}, slice {s_i} has no patches")
                sl_tensor, sl_paths = self._load_slice_tensor(sl)
                tensors_for_stain.append(sl_tensor)
                paths_for_stain.append(sl_paths)

            stain_slices[stain] = tensors_for_stain
            stain_paths[stain] = paths_for_stain

        return {
            "case_id": case_id,
            "stain_slices": stain_slices,
            "stain_paths": stain_paths,
            "label": label,
        }


def case_collate_fn(batch):
    """
    Collate function for MIL dataset.
    Keeps variable-length structures (including stain_paths) as-is;
    use batch_size=1 to avoid padding/masking.
    """
    return batch