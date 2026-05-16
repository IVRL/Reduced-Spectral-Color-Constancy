"""
Gaussian noise injection utilities for reflectance images.

Provides fixed-sigma and SNR-based noise addition, both clipped to keep
reflectance values non-negative.
"""

import numpy as np


def add_gaussian_noise(
        image: np.ndarray,
        sigma: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian noise with a fixed standard deviation to a reflectance image.

    Args:
        image: (N, λ) reflectance pixel array.
        sigma: Standard deviation of the Gaussian noise.
        rng:   Random number generator for reproducibility.

    Returns:
        (N, λ) array of the same dtype as input, with noise added and clipped to >= 0.
    """
    noise = rng.normal(0.0, sigma, size=image.shape).astype(image.dtype)
    return np.clip(image + noise, 0.0, None)


def add_snr_noise(
        image: np.ndarray,
        snr_db: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian noise to a reflectance image at a target SNR (dB).

    SNR is defined relative to the mean signal:
        sigma_noise = mu_signal / 10^(snr_db / 20)

    Args:
        image:  (N, λ) reflectance pixel array.
        snr_db: Target signal-to-noise ratio in dB.
        rng:    Random number generator for reproducibility.

    Returns:
        (N, λ) array of the same dtype as input, with noise added and clipped to >= 0.
    """
    # sigma_noise = mu_signal / 10^(snr_db / 20)
    sigma = image.mean() / (10 ** (snr_db / 20))
    return add_gaussian_noise(image, sigma, rng)
