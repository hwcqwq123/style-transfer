from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# =========================
# Debug hyperparameters
# 先用小一点的参数调试，跑通后再加大
# =========================
STEPS = 1000
PRINT_EVERY = 1
SAVE_DEBUG_EVERY = 100
MAX_SIZE = 384

CONTENT_WEIGHT = 0.5
STYLE_WEIGHT = 3e6
TV_WEIGHT = 5e-6
LR = 0.02

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LayerCfg:
    idx_to_name: Dict[str, str]
    content_layer: str
    style_layers: Dict[str, float]


LAYER_CFG = LayerCfg(
    idx_to_name={
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "21": "conv4_2",
        "28": "conv5_1",
    },
    content_layer="conv4_2",
    style_layers={
        "conv1_1": 1.0,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    },
)


def resize_hw_keep_ratio(w: int, h: int, max_side: int) -> Tuple[int, int]:
    long_side = max(w, h)
    if long_side <= max_side:
        return h, w
    scale = max_side / long_side
    return int(h * scale), int(w * scale)


def build_norm_transform(size_hw: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size_hw),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_tensor_image(path: Path, *, max_side: int, force_hw: Tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    size_hw = resize_hw_keep_ratio(w, h, max_side=max_side) if force_hw is None else force_hw
    tfm = build_norm_transform(size_hw)
    return tfm(img).unsqueeze(0)


def denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    y = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    y = y * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def save_tensor_image(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = denorm_to_uint8(x)
    Image.fromarray(arr).save(str(path), format="JPEG", quality=95)


def get_vgg(device: torch.device) -> torch.nn.Module:
    print("[Adam] Loading VGG19...", flush=True)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    print("[Adam] VGG19 loaded.", flush=True)
    return vgg.to(device)


def extract_features(x: torch.Tensor, vgg: torch.nn.Module, cfg: LayerCfg) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}
    h = x
    for name, layer in vgg._modules.items():
        h = layer(h)
        if name in cfg.idx_to_name:
            feats[cfg.idx_to_name[name]] = h
    return feats


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    _, c, h, w = feat.shape
    f = feat.view(c, h * w)
    g = f @ f.t()
    return g / (c * h * w)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw


def run_adam(content_path, style_path, output_path):
    content_path = Path(content_path)
    style_path = Path(style_path)
    output_path = Path(output_path)

    debug_dir = output_path.parent / "adam_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("[Adam] Start style transfer", flush=True)
    print(f"[Adam] content_path = {content_path}", flush=True)
    print(f"[Adam] style_path   = {style_path}", flush=True)
    print(f"[Adam] output_path  = {output_path}", flush=True)
    print(
        f"[Adam] STEPS={STEPS}, MAX_SIZE={MAX_SIZE}, LR={LR}, "
        f"CONTENT_WEIGHT={CONTENT_WEIGHT}, STYLE_WEIGHT={STYLE_WEIGHT}, TV_WEIGHT={TV_WEIGHT}",
        flush=True
    )

    if not content_path.exists():
        raise FileNotFoundError(f"content image not found: {content_path}")
    if not style_path.exists():
        raise FileNotFoundError(f"style image not found: {style_path}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Adam] device = {device}", flush=True)

        vgg = get_vgg(device)

        print("[Adam] Loading images...", flush=True)
        content = load_tensor_image(content_path, max_side=MAX_SIZE).to(device)
        style = load_tensor_image(style_path, max_side=MAX_SIZE, force_hw=content.shape[-2:]).to(device)
        print(f"[Adam] content tensor shape = {tuple(content.shape)}", flush=True)
        print(f"[Adam] style tensor shape   = {tuple(style.shape)}", flush=True)

        print("[Adam] Extracting content/style features...", flush=True)
        with torch.no_grad():
            content_feats = extract_features(content, vgg, LAYER_CFG)
            style_feats = extract_features(style, vgg, LAYER_CFG)
            style_grams = {
                k: gram_matrix(v)
                for k, v in style_feats.items()
                if k in LAYER_CFG.style_layers
            }
        print("[Adam] Feature extraction done.", flush=True)

        target = content.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=LR)

        total_start = time.time()
        for step in range(1, STEPS + 1):
            step_start = time.time()

            optimizer.zero_grad(set_to_none=True)
            target_feats = extract_features(target, vgg, LAYER_CFG)

            c_loss = F.mse_loss(
                target_feats[LAYER_CFG.content_layer],
                content_feats[LAYER_CFG.content_layer],
            )

            s_loss = torch.tensor(0.0, device=device)
            for layer, weight in LAYER_CFG.style_layers.items():
                target_gram = gram_matrix(target_feats[layer])
                style_gram = style_grams[layer]
                s_loss = s_loss + weight * F.mse_loss(target_gram, style_gram)

            t_loss = tv_loss(target)
            loss = CONTENT_WEIGHT * c_loss + STYLE_WEIGHT * s_loss + TV_WEIGHT * t_loss

            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            total_time = time.time() - total_start

            if step % PRINT_EVERY == 0 or step == 1:
                print(
                    f"[Adam][Step {step:04d}/{STEPS}] "
                    f"total={loss.item():.6e} "
                    f"content={c_loss.item():.6e} "
                    f"style={s_loss.item():.6e} "
                    f"tv={t_loss.item():.6e} "
                    f"step_time={step_time:.3f}s "
                    f"elapsed={total_time:.3f}s",
                    flush=True
                )

            if step % SAVE_DEBUG_EVERY == 0 or step == STEPS:
                debug_img_path = debug_dir / f"adam_step_{step:04d}.jpg"
                save_tensor_image(target, debug_img_path)
                print(f"[Adam] Saved debug image: {debug_img_path}", flush=True)

        print("[Adam] Saving final output...", flush=True)
        save_tensor_image(target, output_path)
        print(f"[Adam] Final output saved: {output_path}", flush=True)
        print("[Adam] Finished successfully.", flush=True)
        print("=" * 60, flush=True)

    except Exception as e:
        print("[Adam] ERROR occurred!", flush=True)
        print(f"[Adam] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise