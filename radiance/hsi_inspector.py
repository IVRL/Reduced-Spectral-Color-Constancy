"""
HSI Inspector — interactively browse hyperspectral .mat images in a directory.

Hover the mouse over the image to see the per-pixel SPD and CIE 1931 xy
chromaticity update live. Use the arrow keys to cycle through images.

Usage:
    python hsi_inspector.py                          # uses default radiance dir
    python hsi_inspector.py <directory>              # any folder of .mat files
    python hsi_inspector.py --wb                     # white-balance via extracted SPD × D65
    python hsi_inspector.py <directory> --wb
"""

import sys
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from illuminants import load_illuminants, get_daylight_by_CCT
from cie_color_matching_functions import cie_1931_2deg_cmfs
from visualization.render_hyperspectral import convert_spectral_to_cspace, CSpace

DARK_FLOOR    = 239
SENSOR_MAX    = 4095
N_BANDS       = 31
DEFAULT_DIR   = os.path.join(REPO_ROOT,
                             'data_raw', 'kaust', 'radiance')
GT_SPD_PATH   = os.path.join(REPO_ROOT,
                             'datasets', 'real_radiance_data', 'radiance_illuminants.npy')


def load_mat(path: str) -> np.ndarray:
    hsi = scipy.io.loadmat(path)['hsi'].astype(np.float32)
    hsi = hsi[:, :, :N_BANDS]
    hsi = (hsi - DARK_FLOOR) / (SENSOR_MAX - DARK_FLOOR)
    return np.clip(hsi, 0.0, None)


def apply_whitebalance(hsi: np.ndarray, scene_name: str, gt_spds: dict) -> np.ndarray:
    """Divide per-band by extracted illuminant SPD → reflectance estimate."""
    if scene_name not in gt_spds:
        return None
    ill = gt_spds[scene_name].astype(np.float32)
    return hsi / np.maximum(ill, 1e-6)


def spectral_locus_xy(cmfs: np.ndarray) -> np.ndarray:
    xyz   = cmfs
    denom = np.maximum(xyz.sum(axis=1, keepdims=True), 1e-12)
    xy    = xyz[:, :2] / denom
    return np.vstack([xy, xy[0]])


def main(image_dir: str, wb: bool = False) -> None:
    wavelengths, _ = load_illuminants()
    cmfs  = cie_1931_2deg_cmfs()
    locus = spectral_locus_xy(cmfs)

    d65_raw = get_daylight_by_CCT(6504).astype(np.float32)
    d65     = d65_raw / d65_raw.sum()

    gt_spds = np.load(GT_SPD_PATH, allow_pickle=True).item()

    mat_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.mat'))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {image_dir}")
    print(f"Found {len(mat_files)} images in {image_dir}")

    # ── Mutable state shared between callbacks ────────────────────────────────
    state = {'idx': 0, 'hsi': None, 'H': 0, 'W': 0, 'wb': wb, 'raw_hsi': None}

    def load_scene(idx: int):
        name = os.path.splitext(mat_files[idx])[0]
        path = os.path.join(image_dir, mat_files[idx])
        hsi  = load_mat(path)
        state['raw_hsi'] = hsi

        return render_scene(idx, hsi, name)

    def render_scene(idx: int, hsi: np.ndarray, name: str):
        if state['wb']:
            refl = apply_whitebalance(hsi, name, gt_spds)
            if refl is None:
                print(f"  [no extracted SPD for '{name}', showing raw]")
                hsi_spd    = hsi
                hsi_render = hsi
                wb_active  = False
            else:
                hsi_spd    = refl
                hsi_render = refl * d65
                wb_active  = True
        else:
            hsi_spd    = hsi
            hsi_render = hsi
            wb_active  = False

        srgb = np.clip(convert_spectral_to_cspace(hsi_render, CSpace.sRGB), 0.0, 1.0)
        H, W, _ = hsi.shape

        state['hsi'] = hsi_spd
        state['H']   = H
        state['W']   = W

        wb_tag = ' [WB]' if wb_active else ''
        title  = f"[{idx+1}/{len(mat_files)}]  {name}{wb_tag}  |  w=toggle WB  ←→=navigate"
        return srgb, title

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a1a')

    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 1],
                           left=0.04, right=0.97, wspace=0.3)
    ax_img = fig.add_subplot(gs[0])
    ax_spd = fig.add_subplot(gs[1])
    ax_xy  = fig.add_subplot(gs[2])

    for ax in (ax_img, ax_spd, ax_xy):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#cccccc', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    # Image panel — load first scene
    srgb, title = load_scene(0)
    im_handle = ax_img.imshow(srgb, interpolation='nearest')
    ax_img.set_title(title, color='#cccccc', fontsize=9)
    ax_img.axis('off')
    cursor_h = ax_img.axhline(-1, color='white', linewidth=0.6, alpha=0.5)
    cursor_v = ax_img.axvline(-1, color='white', linewidth=0.6, alpha=0.5)

    # SPD panel
    spd_line, = ax_spd.plot(wavelengths, np.zeros(N_BANDS),
                             color='#4fc3f7', linewidth=1.8)
    ax_spd.set_xlim(wavelengths[0], wavelengths[-1])
    ax_spd.set_ylim(0, 1)
    ax_spd.set_xlabel('Wavelength (nm)', color='#aaaaaa', fontsize=8)
    ax_spd.set_ylabel('Intensity',       color='#aaaaaa', fontsize=8)
    ax_spd.set_title('SPD',              color='#cccccc', fontsize=9)
    ax_spd.grid(True, color='#333333', linewidth=0.5)
    coord_text = ax_spd.text(0.98, 0.97, '', transform=ax_spd.transAxes,
                              ha='right', va='top', color='#888888', fontsize=7)

    # Chromaticity panel
    ax_xy.plot(locus[:, 0], locus[:, 1], color='#666666', linewidth=1.0)
    ax_xy.plot(0.3127, 0.3290, '+', color='white', markersize=8,
               markeredgewidth=1.5, label='D65')
    ax_xy.set_xlim(0.0, 0.75)
    ax_xy.set_ylim(0.0, 0.85)
    ax_xy.set_xlabel('x',          color='#aaaaaa', fontsize=8)
    ax_xy.set_ylabel('y',          color='#aaaaaa', fontsize=8)
    ax_xy.set_title('CIE 1931 xy', color='#cccccc', fontsize=9)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, color='#333333', linewidth=0.5)
    ax_xy.legend(fontsize=7, labelcolor='#aaaaaa',
                 facecolor='#222222', edgecolor='#444444')
    xy_dot, = ax_xy.plot([], [], 'o', color='#ff5252', markersize=7, zorder=5)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_move(event):
        if event.inaxes is not ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        hsi = state['hsi']
        if hsi is None:
            return

        px = int(round(event.xdata))
        py = int(round(event.ydata))
        px = np.clip(px, 0, state['W'] - 1)
        py = np.clip(py, 0, state['H'] - 1)

        spd = hsi[py, px]

        spd_line.set_ydata(spd)
        ax_spd.set_ylim(0, max(float(spd.max()) * 1.15, 1e-4))
        coord_text.set_text(f'({px}, {py})')

        XYZ   = spd @ cmfs
        total = XYZ.sum()
        xy    = XYZ[:2] / total if total > 1e-6 else np.array([0.333, 0.333])
        xy_dot.set_data([xy[0]], [xy[1]])

        cursor_h.set_ydata([py, py])
        cursor_v.set_xdata([px, px])

        fig.canvas.draw_idle()

    def refresh(srgb, title):
        im_handle.set_data(srgb)
        ax_img.set_title(title, color='#cccccc', fontsize=9)
        spd_line.set_ydata(np.zeros(N_BANDS))
        ax_spd.set_ylim(0, 1)
        coord_text.set_text('')
        xy_dot.set_data([], [])
        cursor_h.set_ydata([-1, -1])
        cursor_v.set_xdata([-1, -1])
        fig.canvas.draw_idle()

    def on_key(event):
        idx  = state['idx']
        name = os.path.splitext(mat_files[idx])[0]

        if event.key in ('left', 'right'):
            idx = (idx + (1 if event.key == 'right' else -1)) % len(mat_files)
            state['idx'] = idx
            print(f"Loading [{idx+1}/{len(mat_files)}] {mat_files[idx]} …")
            srgb, title = load_scene(idx)
            refresh(srgb, title)

        elif event.key == 'w':
            state['wb'] = not state['wb']
            print(f"White balance {'ON' if state['wb'] else 'OFF'}")
            srgb, title = render_scene(idx, state['raw_hsi'], name)
            refresh(srgb, title)

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event',     on_key)

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', nargs='?', default=DEFAULT_DIR,
                        help='Directory of .mat files (default: data_raw/kaust/radiance)')
    parser.add_argument('--wb', action='store_true',
                        help='White-balance using the pre-extracted illuminant SPDs')
    args = parser.parse_args()
    main(args.directory, wb=args.wb)
