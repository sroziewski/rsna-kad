"""MRI-specific intensity augmentation for one series (TrainConfig.mri_aug, qa/mriaug).

One parameter set per series, shared by every slice/channel of that series; each transform fires
with its own fixed probability (pre-declared, not tuned), in this order:

  bias field  p 0.3  multiply by exp(sum c_ij x^i y^j), i + j <= 3, c ~ U(-0.3, 0.3) over the image
                     in [-1, 1]^2 (the TorchIO RandomBiasField form, order 3)
  gamma       p 0.3  x ** exp(g), g ~ U(-0.3, 0.3)
  blur        p 0.2  Gaussian, sigma ~ U(0, 0.8) px
  noise       p 0.2  additive N(0, s), s ~ U(0, 0.03) of the [0, 1] range

Works on uint8 (..., H, W) arrays and returns uint8 of the same shape (the value range stays
[0, 255]). Intensity only: never moves or mirrors a pixel. Hand-rolled (numpy + cv2) so it runs in
the shared venv and in any other pipeline that imports it.
"""
from __future__ import annotations

import cv2
import numpy as np

P_BIAS, BIAS_COEF, BIAS_ORDER = 0.3, 0.3, 3
P_GAMMA, LOG_GAMMA = 0.3, 0.3
P_BLUR, BLUR_SIGMA = 0.2, 0.8
P_NOISE, NOISE_SIGMA = 0.2, 0.03


def bias_field(h: int, w: int, coefs: np.ndarray, order: int = BIAS_ORDER) -> np.ndarray:
    """exp of a polynomial of total degree <= order over [-1, 1]^2; coefs in (i, j) order."""
    y, x = np.meshgrid(np.linspace(-1.0, 1.0, h, dtype=np.float32), np.linspace(-1.0, 1.0, w, dtype=np.float32),
                       indexing="ij")
    field = np.zeros((h, w), np.float32)
    n = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            field += np.float32(coefs[n]) * x ** i * y ** j
            n += 1
    return np.exp(field)


def n_bias_coefs(order: int = BIAS_ORDER) -> int:
    return (order + 1) * (order + 2) // 2


def sample_params(rng: np.random.Generator) -> dict:
    """The random draws for one series (fixed draw order, so a seed fixes the result)."""
    p = {}
    if rng.random() < P_BIAS:
        p["bias"] = rng.uniform(-BIAS_COEF, BIAS_COEF, n_bias_coefs())
    if rng.random() < P_GAMMA:
        p["gamma"] = float(np.exp(rng.uniform(-LOG_GAMMA, LOG_GAMMA)))
    if rng.random() < P_BLUR:
        p["blur"] = float(rng.uniform(0.0, BLUR_SIGMA))
    if rng.random() < P_NOISE:
        p["noise"] = float(rng.uniform(0.0, NOISE_SIGMA))
    return p


def apply_params(img_u8: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    if not p:
        return img_u8
    shape = img_u8.shape
    h, w = shape[-2:]
    x = img_u8.reshape(-1, h, w).astype(np.float32) / 255.0
    if "bias" in p:
        x = x * bias_field(h, w, p["bias"])[None]
    if "gamma" in p:
        x = np.clip(x, 0.0, 1.0) ** np.float32(p["gamma"])
    if "blur" in p and p["blur"] > 0.05:                       # below that the kernel is an identity
        for i in range(x.shape[0]):
            x[i] = cv2.GaussianBlur(x[i], (0, 0), p["blur"])
    if "noise" in p:
        x = x + rng.normal(0.0, p["noise"], x.shape).astype(np.float32)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(shape)


def mri_augment(img_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One series (..., H, W) uint8 -> same shape uint8, one parameter set for all of it."""
    return apply_params(img_u8, sample_params(rng), rng)
