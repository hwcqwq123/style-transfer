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

DEFAULT_STEPS = 1000
DEFAULT_PRINT_EVERY = 1
DEFAULT_SAVE_DEBUG_EVERY = 100
DEFAULT_MAX_SIZE = 384

DEFAULT_CONTENT_WEIGHT = 0.5
DEFAULT_STYLE_WEIGHT = 3e6
DEFAULT_TV_WEIGHT = 1e-5

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


def get_param(params: dict | None, key: str, default, cast_func):
    if not params or key not in params:
        return default
    try:
        return cast_func(params[key])
    except Exception:
        return default


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
    print("[LBFGS] Loading VGG19...", flush=True)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    print("[LBFGS] VGG19 loaded.", flush=True)
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


def run_lbfgs(content_path, style_path, output_path, params=None):
    content_path = Path(content_path)
    style_path = Path(style_path)
    output_path = Path(output_path)
    params = params or {}

    STEPS = max(1, get_param(params, "steps", DEFAULT_STEPS, int))
    PRINT_EVERY = max(1, get_param(params, "print_every", DEFAULT_PRINT_EVERY, int))
    SAVE_DEBUG_EVERY = max(1, get_param(params, "save_debug_every", DEFAULT_SAVE_DEBUG_EVERY, int))
    MAX_SIZE = max(64, get_param(params, "max_size", DEFAULT_MAX_SIZE, int))
    CONTENT_WEIGHT = max(0.0, get_param(params, "content_weight", DEFAULT_CONTENT_WEIGHT, float))
    STYLE_WEIGHT = max(0.0, get_param(params, "style_weight", DEFAULT_STYLE_WEIGHT, float))
    TV_WEIGHT = max(0.0, get_param(params, "tv_weight", DEFAULT_TV_WEIGHT, float))

    debug_dir = output_path.parent / "lbfgs_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("[LBFGS] Start style transfer", flush=True)
    print(f"[LBFGS] params       = {params}", flush=True)
    print(f"[LBFGS] content_path = {content_path}", flush=True)
    print(f"[LBFGS] style_path   = {style_path}", flush=True)
    print(f"[LBFGS] output_path  = {output_path}", flush=True)
    print(
        f"[LBFGS] STEPS={STEPS}, PRINT_EVERY={PRINT_EVERY}, SAVE_DEBUG_EVERY={SAVE_DEBUG_EVERY}, "
        f"MAX_SIZE={MAX_SIZE}, CONTENT_WEIGHT={CONTENT_WEIGHT}, "
        f"STYLE_WEIGHT={STYLE_WEIGHT}, TV_WEIGHT={TV_WEIGHT}",
        flush=True
    )

    if not content_path.exists():
        raise FileNotFoundError(f"content image not found: {content_path}")
    if not style_path.exists():
        raise FileNotFoundError(f"style image not found: {style_path}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LBFGS] device = {device}", flush=True)

        vgg = get_vgg(device)

        print("[LBFGS] Loading images...", flush=True)
        content = load_tensor_image(content_path, max_side=MAX_SIZE).to(device)
        style = load_tensor_image(style_path, max_side=MAX_SIZE, force_hw=content.shape[-2:]).to(device)
        print(f"[LBFGS] content tensor shape = {tuple(content.shape)}", flush=True)
        print(f"[LBFGS] style tensor shape   = {tuple(style.shape)}", flush=True)

        print("[LBFGS] Extracting content/style features...", flush=True)
        with torch.no_grad():
            content_feats = extract_features(content, vgg, LAYER_CFG)
            style_feats = extract_features(style, vgg, LAYER_CFG)
            style_grams = {
                k: gram_matrix(v)
                for k, v in style_feats.items()
                if k in LAYER_CFG.style_layers
            }
        print("[LBFGS] Feature extraction done.", flush=True)

        target = content.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([target], max_iter=STEPS, line_search_fn="strong_wolfe")

        state = {
            "step": 0,
            "start_time": time.time(),
        }

        def closure():
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

            state["step"] += 1
            step = state["step"]
            elapsed = time.time() - state["start_time"]

            if step % PRINT_EVERY == 0 or step == 1:
                print(
                    f"[LBFGS][Step {step:04d}/{STEPS}] "
                    f"total={loss.item():.6e} "
                    f"content={c_loss.item():.6e} "
                    f"style={s_loss.item():.6e} "
                    f"tv={t_loss.item():.6e} "
                    f"elapsed={elapsed:.3f}s",
                    flush=True
                )

            if step % SAVE_DEBUG_EVERY == 0 or step == STEPS:
                debug_img_path = debug_dir / f"lbfgs_step_{step:04d}.jpg"
                save_tensor_image(target, debug_img_path)
                print(f"[LBFGS] Saved debug image: {debug_img_path}", flush=True)

            return loss

        print("[LBFGS] Running optimizer...", flush=True)
        optimizer.step(closure)

        print("[LBFGS] Saving final output...", flush=True)
        save_tensor_image(target, output_path)
        print(f"[LBFGS] Final output saved: {output_path}", flush=True)
        print("[LBFGS] Finished successfully.", flush=True)
        print("=" * 60, flush=True)

    except Exception as e:
        print("[LBFGS] ERROR occurred!", flush=True)
        print(f"[LBFGS] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise