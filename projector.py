"""
Projection wrappers for the CBC pipeline.

Defines the Projector protocol and three concrete implementations:
  - SklearnProjector: wraps a fitted scikit-learn model (PCA, NMF, LDA)
  - SensorProjector:  integrates spectra against RGB camera sensitivities
  - RandomProjector:  random linear projection with a fixed seed
"""

import numpy as np
from typing import Protocol, runtime_checkable

from experiment_settings import CameraRGBMode as M


@runtime_checkable
class Projector(Protocol):
    n_components: int

    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def basis(self) -> np.ndarray: ...
    def basis_labels(self) -> list[str]: ...


class SklearnProjector:
    """Wraps a fitted scikit-learn projector (PCA, NMF, LDA)."""

    def __init__(self, skl_projection):
        self.skl_projection = skl_projection
        self.n_components = skl_projection.n_components

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.skl_projection.transform(X)

    def basis(self) -> np.ndarray:
        """Return projection components, shape (n_components, n_features)."""
        return self.skl_projection.components_

    def basis_labels(self) -> list[str]:
        return [f"Component {i + 1}" for i in range(self.skl_projection.n_components_)]


class SensorProjector:
    """
    Projects spectra to RGB (or a derived 2D chromaticity) via camera sensitivity integration.

    Supported modes:
        RGB:   raw sensor response (3D)
        RG:    rg chromaticity — r = R/(R+G+B), g = G/(R+G+B)
        RG3:   cube-root of rg chromaticity (2D)
        UV:    ratio chromaticity — [R/G, B/G] (2D)
        UV3:   cube-root UV (2D)
        LOGUV: log UV (2D)
    """

    def __init__(self, sensitivities: np.ndarray, delta_nm: float = 10.0, mode: M = M.RGB):
        """
        Args:
            sensitivities: (Nλ, 3) array — columns are R, G, B spectral sensitivities.
            delta_nm:      Wavelength step in nm (used to scale the integration kernel).
            mode:          Output mode (CameraRGBMode).
        """
        self.sensitivities = sensitivities.astype(np.float32)
        self.K = self.sensitivities * delta_nm
        self.mode = mode
        self.n_components = 3 if mode == M.RGB else 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Integrate spectra against the camera sensitivity kernel."""
        rgb = X @ self.K

        if self.mode == M.RG:
            denom = rgb.sum(axis=1, keepdims=True)
            return np.divide(rgb, denom, where=denom != 0)[:, :2]

        if self.mode == M.RG3:
            denom = rgb.sum(axis=1, keepdims=True)
            return np.divide(rgb, denom, where=denom != 0)[:, :2] ** (1 / 3)

        if self.mode in (M.UV, M.UV3, M.LOGUV):
            g = rgb[:, 1]
            mask = g > 1e-12
            uv = rgb[mask][:, [0, 2]] / g[mask, None]  # [R/G, B/G]
            if self.mode == M.UV3:
                uv = np.sign(uv) * np.abs(uv) ** (1 / 3)
            elif self.mode == M.LOGUV:
                uv = np.log(uv)
            return uv

        return rgb  # M.RGB

    def basis(self) -> np.ndarray:
        """Return sensor sensitivities as (3, Nλ) to match PCA component layout."""
        return self.sensitivities.T

    def basis_labels(self) -> list[str]:
        return ["Sensor R", "Sensor G", "Sensor B"]


class RandomProjector:
    """
    Random linear spectral projector with a fixed seed.

    Always generates MAX_COMPONENTS random basis vectors and exposes the
    first n_components, so increasing n_components with the same seed gives
    a nested sequence of projections.
    """

    # The full basis is generated once from MAX_COMPONENTS vectors. n_components
    # selects a prefix. Changing MAX_COMPONENTS shifts all basis vectors and
    # breaks reproducibility across runs with the same seed.
    MAX_COMPONENTS = 10
    N_WAVELENGTHS  = 31

    def __init__(self, n_components: int, seed: int = 0):
        if not (1 <= n_components <= self.MAX_COMPONENTS):
            raise ValueError(f"n_components must be in [1, {self.MAX_COMPONENTS}]")

        self.n_components = int(n_components)
        rng = np.random.default_rng(seed)

        # Generate the full basis once; use a prefix of it for the active projection.
        self._full_sensitivities = rng.uniform(
            -1.0, 1.0, size=(self.N_WAVELENGTHS, self.MAX_COMPONENTS)
        ).astype(np.float32)
        self.sensitivities = self._full_sensitivities[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.sensitivities

    def basis(self) -> np.ndarray:
        return self.sensitivities.T  # (n_components, 31)

    def basis_labels(self) -> list[str]:
        return [f"Random {i + 1}" for i in range(self.n_components)]
