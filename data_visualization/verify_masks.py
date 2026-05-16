"""
Cycle through KAUST hyperspectral images with masks overlaid in red.
Use left/right arrow keys (or A/D) to navigate. Press Q to quit.
"""
import os
import sys

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.render_hyperspectral import convert_spectral_to_cspace, CSpace

H5_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw", "kaust", "h5")
MASK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw", "kaust", "masks")

h5_files = sorted(f for f in os.listdir(H5_DIR) if f.endswith(".h5"))


def load_image_and_mask(idx: int):
    fname = h5_files[idx]
    base = fname[:-3]
    mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")

    with h5py.File(os.path.join(H5_DIR, fname), "r") as f:
        raw = next(iter(f.values()))  # (C, H, W)
        spectral = raw[:31].transpose(1, 2, 0).astype(np.float32)  # (H, W, 31)

    spectral /= spectral.max() if spectral.max() > 0 else 1.0
    srgb = convert_spectral_to_cspace(spectral, CSpace.sRGB)  # (H, W, 3)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0  # (H, W) bool

    # Overlay mask as red tint
    overlay = srgb.copy()
    overlay[mask, 0] = 0.3
    overlay[mask, 1] *= 0.3
    overlay[mask, 2] *= 0.3

    return overlay, mask, base


state = {"idx": 0}

fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title("KAUST Mask Verification")
plt.tight_layout()


def draw(idx: int):
    overlay, mask, name = load_image_and_mask(idx)
    ax.clear()
    ax.imshow(np.clip(overlay, 0, 1))
    n_foreground = mask.sum()
    ax.set_title(f"[{idx+1}/{len(h5_files)}]  {name}   foreground px: {n_foreground}", fontsize=10)
    ax.axis("off")
    fig.canvas.draw()


def on_key(event):
    idx = state["idx"]
    if event.key in ("right", "d"):
        idx = (idx + 1) % len(h5_files)
    elif event.key in ("left", "a"):
        idx = (idx - 1) % len(h5_files)
    elif event.key in ("q", "escape"):
        plt.close()
        return
    else:
        return
    state["idx"] = idx
    draw(idx)


fig.canvas.mpl_connect("key_press_event", on_key)
draw(0)
print(f"Loaded {len(h5_files)} images. Use left/right arrows to navigate, Q to quit.")
plt.show()
