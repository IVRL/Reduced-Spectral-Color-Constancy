"""
Grid search over CBC experiment parameters.

Edit the `grid` dict to define the parameter combinations to sweep, then run:

    python run_experiment_grid.py

Results are appended to results/cbc_runs.csv after each experiment.
"""

import time
from itertools import product
from tqdm import tqdm
from illuminants import ILL_TEST_SET, ILL_TRAIN_SET
from experiment_settings import DimRedMethod
from run_experiment import run_experiment


if __name__ == '__main__':

    grid = {
        "dim_red_methods":      [DimRedMethod.ILL_PCA, DimRedMethod.PCA, DimRedMethod.CAM],
        "num_components":       [2, 3, 4],
        "seeds":                [43],
        "histogram_bin_counts": [30],
        "snr_db_levels":        [None], # None for clean
    }

    combinations = list(product(
        grid["dim_red_methods"],
        grid["num_components"],
        grid["seeds"],
        grid["histogram_bin_counts"],
        grid["snr_db_levels"],
    ))

    for method, num_components, seed, num_bins, snr_db in tqdm(combinations, desc="Running experiments"):
        settings = {
            "experiment_name": "",
            "run_notes": "",
            "dim_red_method": method,
            "test_illuminant_names": ILL_TEST_SET,
            "projection_illuminant_names": ILL_TRAIN_SET,
            "num_components": num_components,
            "random_projection_seed": seed,
            "threshold": True,
            "histogram_bin_count": num_bins,
            "snr_db": snr_db,
        }

        snr_str = f"{snr_db}dB" if snr_db is not None else "clean"
        tqdm.write(f"Running: method={method.value}  components={num_components}  seed={seed}  bins={num_bins}  snr={snr_str}")
        t0 = time.time()
        run_experiment(settings, test_on_radiance=False)
        tqdm.write(f"Done in {time.time() - t0:.1f}s\n")
