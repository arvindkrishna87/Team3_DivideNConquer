#!/usr/bin/env python3
"""
precompute_pooled_features_uni2.py

Precompute frozen UNI2-h embeddings for variable-sized patch PNGs.

This does not replace precompute_pooled_features.py. KimiaNet vectors are 4096-d
and cannot be converted into UNI2 vectors.

For each valid patch:
  embeddings_dir/<basename>.pt is a 1D tensor of shape (1536,).

Preprocessing follows UNI2 normalization and does not resize or crop to 224.
dynamic_img_size=True lets the ViT take the original spatial size. Height and
width are only padded so both are multiples of the patch size (14).
"""

import os
import time
import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import timm


EMBED_DIM = 1536
PATCH_SIZE = 14
UNI2_MEAN = (0.485, 0.456, 0.406)
UNI2_STD = (0.229, 0.224, 0.225)

try:
    from timm.layers import SwiGLUPacked
except ImportError:
    from timm.models.layers import SwiGLUPacked


def uni2_timm_kwargs():
    """Architecture kwargs from the MahmoodLab/UNI2-h model card."""
    return {
        "img_size": 224,
        "patch_size": PATCH_SIZE,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": EMBED_DIM,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


class PatchPathDataset(Dataset):
    """
    Returns (fname, tensor, status) for each image.

    status is "ok" or "bad_read". Images are not resized. Spatial size is padded
    to a multiple of PATCH_SIZE after conversion to a tensor and before normalization.
    """

    def __init__(self, patches_dir: str, filenames: List[str]):
        self.patches_dir = patches_dir
        self.filenames = filenames
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=UNI2_MEAN, std=UNI2_STD)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[str, Optional[torch.Tensor], str]:
        fname = self.filenames[idx]
        path = os.path.join(self.patches_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return fname, None, "bad_read"

        x = self.to_tensor(img)
        x = pad_to_patch_multiple(x, PATCH_SIZE)
        x = self.normalize(x)
        return fname, x, "ok"


def pad_to_patch_multiple(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad the right and bottom so H and W are multiples of the ViT patch size.

    x is a float tensor in [0, 1] with shape (C, H, W). Padding uses 0, which
    becomes the normalized mean color after UNI2 normalization.
    """
    _, height, width = x.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=0.0)


def collate_keep_list(batch):
    """Keep items as a list because tensors may have different H x W."""
    return batch


def _unwrap_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_uni2(ckpt_path: str, device: str) -> torch.nn.Module:
    """Load frozen UNI2-h. A local checkpoint is used when ckpt_path is set."""
    kwargs = uni2_timm_kwargs()
    if ckpt_path:
        model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(_unwrap_state_dict(ckpt), strict=True)
        print(f"[INFO] Loaded UNI2-h weights from {ckpt_path}")
    else:
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **kwargs,
        )
        print("[INFO] Loaded UNI2-h weights from MahmoodLab/UNI2-h")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def class_token(features: torch.Tensor) -> torch.Tensor:
    """Return a (1536,) class token from UNI2's forward output."""
    if features.dim() == 2:
        token = features[0]
    elif features.dim() == 3:
        token = features[0, 0]
    else:
        raise RuntimeError(f"Unexpected UNI2 output shape {tuple(features.shape)}")
    if token.numel() != EMBED_DIM:
        raise RuntimeError(
            f"Expected a {EMBED_DIM}-d UNI2 embedding, got shape {tuple(token.shape)}"
        )
    return token


def list_pngs_in_dir(patches_dir: str) -> List[str]:
    fnames = [f for f in os.listdir(patches_dir) if f.lower().endswith(".png")]
    fnames.sort()
    return fnames


def out_path_for_fname(embeddings_dir: str, fname: str) -> str:
    out_name = os.path.splitext(fname)[0] + ".pt"
    return os.path.join(embeddings_dir, out_name)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patches_dir",
        default="/projects/e32998/patches_benign_split",
        help="Folder with the original PNG patches",
    )
    ap.add_argument(
        "--embeddings_dir",
        default="/home/cny2467/uni2_embeddings_1536",
        help="Where to save 1536-d .pt features",
    )
    ap.add_argument(
        "--uni2_ckpt",
        default="",
        help="Local pytorch_model.bin. Empty uses the Hugging Face hub.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Used only for loading items; the forward pass is per patch",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)

    print(f"[INFO] patches_dir:      {args.patches_dir}")
    print(f"[INFO] embeddings_dir:   {args.embeddings_dir}")
    print(f"[INFO] uni2_ckpt:        {args.uni2_ckpt or 'MahmoodLab/UNI2-h'}")
    print(f"[INFO] device:           {args.device}")
    print(f"[INFO] batch_size:       {args.batch_size}")
    print(f"[INFO] num_workers:      {args.num_workers}")
    print(f"[INFO] overwrite:        {args.overwrite}")
    print(f"[INFO] dtype:            {args.dtype}")
    print(f"[INFO] embed_dim:        {EMBED_DIM}")
    print("[INFO] resizing:         disabled")
    print(f"[INFO] padding:          right/bottom to a multiple of {PATCH_SIZE}")
    print("[INFO] forward pass:     one frozen patch at a time")

    filenames = list_pngs_in_dir(args.patches_dir)
    total_png = len(filenames)
    print(f"[INFO] Found {total_png} PNG files.")

    model = load_uni2(args.uni2_ckpt, device=args.device)

    ds = PatchPathDataset(args.patches_dir, filenames)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_list,
        persistent_workers=(args.num_workers > 0),
    )

    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    seen_total = 0
    saved_total = 0
    skipped_existing = 0
    skipped_bad = 0

    start_time = time.time()

    with torch.inference_mode():
        for step, batch in enumerate(dl, start=1):
            for fname, x, status in batch:
                seen_total += 1

                if status == "bad_read":
                    skipped_bad += 1
                    continue

                out_path = out_path_for_fname(args.embeddings_dir, fname)
                if (not args.overwrite) and os.path.exists(out_path):
                    skipped_existing += 1
                    continue

                x = x.unsqueeze(0).to(args.device, non_blocking=True)
                token = class_token(model(x))
                torch.save(token.to(dtype=save_dtype).cpu(), out_path)
                saved_total += 1

            if args.log_every > 0 and (step % args.log_every == 0 or seen_total >= total_png):
                elapsed = time.time() - start_time
                rate = seen_total / elapsed if elapsed > 0 else 0.0
                remaining = (total_png - seen_total) / rate if rate > 0 else 0.0
                print(
                    f"[PROGRESS] step={step} seen={seen_total}/{total_png} "
                    f"saved={saved_total} skipped_existing={skipped_existing} "
                    f"skipped_bad={skipped_bad} "
                    f"rate={rate:.2f} patches/sec ETA={format_eta(remaining)}"
                )

    total_elapsed = time.time() - start_time
    print("[DONE]")
    print(f"[SUMMARY] total_png={total_png}")
    print(f"[SUMMARY] saved={saved_total}")
    print(f"[SUMMARY] skipped_existing={skipped_existing}")
    print(f"[SUMMARY] skipped_bad={skipped_bad}")
    print(f"[SUMMARY] embeddings_dir={args.embeddings_dir}")
    print(f"[SUMMARY] total_time={format_eta(total_elapsed)}")
    print(f"Note: saved tensors are length {EMBED_DIM} (frozen UNI2-h class token).")


if __name__ == "__main__":
    main()
