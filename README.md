# Low-Dimensional Hyperspectral Color-by-Correlation

This repository contains the official implementation of **"Low-Dimensional Hyperspectral Color-by-Correlation for Illuminant Estimation"**, to be presented at IEEE ICIP 2026.
Paper can be found here: https://arxiv.org/abs/2605.13306v1

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib colour-science h5py openpyxl xlrd tqdm opencv-python
```

## Data Setup

### KAUST Hyperspectral Dataset

The KAUST spectral reflectance images (`.h5` files) must be obtained separately and placed in:


```
data_raw/kaust/h5/
```

Masks are already included in `data_raw/kaust/masks/`.

At the time of writing, the dataset can be downloaded from: https://repository.kaust.edu.sa/items/12d8306d-8fd9-43e8-a569-dbc364454a36

### Generating Preprocessed Datasets

Different stages of the pipeline use different downscale levels:

- **ds2** — used for evaluation (`test_model.py`, `grey_world_baseline.py`). More pixels gives more accurate metrics.
- **ds4** — used for precomputed radiances (`precompute_radiances.py`). Balances pixel count against RAM and disk usage.
- **ds8** — used when fitting NMF and LDA projectors. These methods load the full training set into memory at once (no incremental API), so heavy downscaling is needed. IncrementalPCA can use ds4 since it processes one image at a time.

Generate the datasets at all three levels:

```bash
python generate_datasets.py              # ds2 (default)
python generate_datasets.py --downscale 4
python generate_datasets.py --downscale 8
```

Output is saved to `datasets/`:

```
datasets/
  X_train_ds{N}.npy   # images  (N_images, H, W, 31)
  M_train_ds{N}.npy   # masks   (N_images, H, W)
  X_test_ds{N}.npy
  M_test_ds{N}.npy
```

Then precompute the per-illuminant radiance pixel files used during model building:

```bash
python precompute_radiances.py           # ds4 (default)
```

Output is saved to `datasets/radiances_ds4/`, one `.npy` file per illuminant.

### Projection Models

Pre-fitted projection models (PCA, NMF, LDA, ILL-PCA) for n_components 1–6 are included under `models/`. To regenerate them:

```bash
python fit_projections.py all            # fits all methods for n_components 1–6 at ds8
python fit_projections.py pca --n_components 3   # individual
```

---

## Running Experiments

### Single experiment

```python
from run_experiment import run_experiment
from experiment_settings import DimRedMethod

run_experiment({
    "experiment_name":            "my_run",
    "dim_red_method":             DimRedMethod.ILL_PCA,
    "num_components":             3,
    "histogram_bin_count":        30,
    "threshold":                  True,
})
```

Or edit and run the grid script directly:

```bash
python run_experiment_grid.py
```

Results are appended to `results/cbc_runs.csv`.

### Settings reference

| Setting | Default   | Description                                                                                      |
|---|-----------|--------------------------------------------------------------------------------------------------|
| `dim_red_method` | `ILL_PCA` | Projection method: `NMF`, `ILL_NMF`, `PCA`, `ILL_PCA`, `LDA`, `CAM`, `RAND`                      |
| `num_components` | `3`       | Projected dimensionality                                                                         |
| `histogram_bin_count` | `30`      | Bins per dimension in the CBC histogram                                                          |
| `threshold` | `False`   | Make histograms into binary presence indicators as in original CbC, (False in paper experiments) |
| `snr_db` | `None`    | Add sensor noise to test images at this SNR (dB); `None` = clean                                 |
| `test_illuminant_names` | all       | List of candidate illuminant names to test against                                               |
| `projection_illuminant_names` | all       | Illuminants used to fit the projection                                                           |

### Grey-world baseline

```bash
python grey_world_baseline.py
```

---

## Project Structure

```
illuminants.py             # CIE illuminant loader (daylight, fluorescent, A, LED)
cie_color_matching_functions.py  # CIE 1931 2° CMFs on the 400–700 nm grid
camera_sensitivity.py      # RGB camera spectral sensitivities (CamSpec database)

generate_datasets.py       # raw .h5 → train/test .npy
kaust_dataset.py           # dataset loaders used during training and evaluation
precompute_radiances.py    # pre-multiply pixels by illuminant SPDs for faster histogram construction
fit_projections.py         # fit and save PCA / NMF / LDA projectors

cbc_model.py               # SimpleCBC model (histogram-based illuminant estimation)
projector.py               # projection wrappers (PCA, NMF, LDA, camera, random)
create_models.py           # model factory — assembles projector + CBC model

experiment_settings.py     # DimRedMethod / CameraRGBMode enums + settings validation
experiment_store.py        # CSV-based result storage with typed schema
run_experiment.py          # single-experiment entry point
run_experiment_grid.py     # run experiments over factorial parameter combinations
test_model.py              # evaluation on KAUST synthetic test set
grey_world_baseline.py     # spectral grey-world baseline for comparison
noise.py                   # Gaussian / SNR noise injection utilities

radiance/                  # evaluation on real captured radiance images (optional)
  extract_illuminants.py   # estimate illuminant SPDs from whiteboard masks
  generate_dataset.py      # process radiance .mat files into .npy dataset
  test_model.py            # evaluate a CBC model on the radiance test set
  hsi_inspector.py         # interactive per-pixel SPD / chromaticity viewer

data_visualization/        # Jupyter notebooks for analysis and results
visualization/             # sRGB rendering utilities
```

## Illuminants

All experiments use CIE standard illuminants sampled at 400–700 nm (10 nm steps, 31 bands). The train/test split is defined in `illuminants.py` as `ILL_TRAIN_SET` / `ILL_TEST_SET`.

Families included: daylight (D series), fluorescent (F1–F12), CIE Illuminant A, and CIE LED illuminants (B, BH, RGB, V series).
