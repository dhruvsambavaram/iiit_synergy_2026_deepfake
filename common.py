"""
common.py -- FROZEN. Shared constants, image loading and the robustness corruption suite.
Do not edit during the research loop (evaluate.py verifies its hash).
"""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO = Path(__file__).resolve().parent
SPLITS = REPO / "data_splits"
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed(image_id: str, name: str) -> int:
    return int(hashlib.md5(f"{image_id}|{name}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- corruptions
def _jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return load_rgb(buf)


def _webp(img, q):
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=q); buf.seek(0)
    return load_rgb(buf)


def _downup(img, s):
    w, h = img.size
    small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _noise(img, sigma, rng):
    a = np.asarray(img).astype(np.float32)
    a = a + rng.normal(0, sigma * 255, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _motion(img, length=9):
    a = np.asarray(img).astype(np.float32)
    pad = length // 2
    p = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = sum(p[:, i:i + a.shape[1]] for i in range(length)) / length
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _gamma(img, g):
    a = np.asarray(img).astype(np.float32) / 255.0
    return Image.fromarray(np.clip((a ** g) * 255, 0, 255).astype(np.uint8))


def _center_crop(img, frac):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BICUBIC)


def _social(img):  # downscale -> recompress -> upscale (typical re-share pipeline)
    w, h = img.size
    small = img.resize((int(w * 0.75), int(h * 0.75)), Image.BILINEAR)
    return _jpeg(small, 60).resize((w, h), Image.BILINEAR)


def _color(img):
    return ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.25)).enhance(0.75)


def _cutout(img, rng):  # organiser-style occlusion box (gray/black/white), seen in train data
    a = np.asarray(img).copy(); h, w = a.shape[:2]
    s = int(rng.uniform(0.15, 0.25) * w); x, y = rng.integers(0, w - s), rng.integers(0, h - s)
    a[y:y + s, x:x + s] = rng.choice([0, 128, 255]); return Image.fromarray(a)


def _pixelate(img, s=0.25):
    w, h = img.size
    return img.resize((int(w * s), int(h * s)), Image.NEAREST).resize((w, h), Image.NEAREST)


# name -> (fn(img, rng), group). "seen" types MAY be used as training augmentation.
# "unseen" types MUST NOT be used (or closely imitated) in train.py: they measure true generalization.
CORRUPTIONS = {
    "jpeg50":     (lambda im, r: _jpeg(im, 50), "seen"),
    "jpeg30":     (lambda im, r: _jpeg(im, 30), "seen"),
    "gblur1":     (lambda im, r: im.filter(ImageFilter.GaussianBlur(1.0)), "seen"),
    "gblur2":     (lambda im, r: im.filter(ImageFilter.GaussianBlur(2.0)), "seen"),
    "noise02":    (lambda im, r: _noise(im, 0.02, r), "seen"),
    "noise05":    (lambda im, r: _noise(im, 0.05, r), "seen"),
    "downup50":   (lambda im, r: _downup(im, 0.5), "seen"),
    "color":      (lambda im, r: _color(im), "seen"),
    "cutout":     (lambda im, r: _cutout(im, r), "seen"),
    "pixelate":   (lambda im, r: _pixelate(im), "seen"),
    "gray":       (lambda im, r: im.convert("L").convert("RGB"), "seen"),
    "webp30":     (lambda im, r: _webp(im, 30), "unseen"),
    "motion9":    (lambda im, r: _motion(im, 9), "unseen"),
    "gamma15":    (lambda im, r: _gamma(im, 1.5), "unseen"),
    "posterize4": (lambda im, r: ImageOps.posterize(im, 4), "unseen"),
    "crop85":     (lambda im, r: _center_crop(im, 0.85), "unseen"),
    "median3":    (lambda im, r: im.filter(ImageFilter.MedianFilter(3)), "unseen"),
    "social":     (lambda im, r: _social(im), "unseen"),
}
VARIANTS = ["clean"] + list(CORRUPTIONS)


def apply_variant(img: Image.Image, variant: str, image_id: str) -> Image.Image:
    if variant == "clean":
        return img
    fn, _ = CORRUPTIONS[variant]
    return fn(img, np.random.default_rng(_seed(image_id, variant)))
