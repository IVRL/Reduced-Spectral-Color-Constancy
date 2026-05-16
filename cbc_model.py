"""
CBC model definition.

Defines the abstract Model base class and the concrete SimpleCBC implementation.
SimpleCBC estimates the illuminant of an image by projecting its pixels into a
low-dimensional space, building a histogram of the projected values, and computing
the dot product against pre-built per-illuminant histograms (log-likelihood proxy).
"""

import pickle
from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np
import uuid

from projector import Projector


class Model(ABC):
    """Abstract base class for all illuminant estimation models."""

    def __init__(self):
        self.id = uuid.uuid4()

    @abstractmethod
    def estimate_illuminant(self, image: np.ndarray, threshold: bool) -> np.ndarray:
        """
        Estimate the log-likelihood of each candidate illuminant given an image.

        Args:
            image:     (N, 31) array of L1-normalised pixel spectra.
            threshold: If True, clip histogram counts to 1 before scoring
                       (presence/absence rather than frequency).

        Returns:
            (K,) array of scores — higher means more likely. Not normalised.
        """

    @abstractmethod
    def save(self) -> str:
        """Serialise the model to disk and return the save path."""

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a model from a pickle file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj.path = path
        return obj


class SimpleCBC(Model):
    """
    Histogram-based color-by-correlation illuminant estimator.

    At construction time the model receives pre-built per-illuminant histograms
    (combined_hists). At inference time the image pixels are projected, binned
    into a histogram using the same edges, and scored against each illuminant
    histogram via a dot product.
    """

    def __init__(self, model_dict: Mapping):
        super().__init__()
        if not isinstance(model_dict, Mapping):
            raise TypeError("SimpleCBC expects a dict-like object; got a direct instance.")

        required = {"projector", "combined_hists", "edges"}
        missing = required - set(model_dict.keys())
        if missing:
            raise KeyError(f"Model dict missing keys: {sorted(missing)}")

        self.projector: Projector = model_dict["projector"]
        self.combined_hists: np.ndarray = np.ascontiguousarray(model_dict["combined_hists"])
        self.edges = model_dict["edges"]
        self.path = f"models/cbc_histograms/{self.id}.pickle"

    def estimate_illuminant(self, image: np.ndarray, threshold: bool) -> np.ndarray:
        """
        Score each candidate illuminant against the image's pixel distribution.

        Args:
            image:     (N, 31) array of L1-normalised pixel spectra.
            threshold: If True, clip histogram counts to 1 before scoring
                       (presence/absence rather than frequency).

        Returns:
            (K,) array of scores — higher means more likely. Not normalised.
        """
        projected = self.projector.transform(image)

        H_image, _ = np.histogramdd(
            projected[:, :self.projector.n_components],
            bins=self.edges,
        )
        H_image = H_image.astype(np.float32, copy=False).ravel()

        if threshold:
            # Treat each bin as occupied or not, removes the influence of
            # dominant chromaticities that would otherwise dominate the dot product.
            H_image = np.minimum(H_image, 1.0)

        # combined_hists: (K, B^d') where K = number of illuminants.
        # Dot product gives a log-likelihood proxy for each illuminant.
        return self.combined_hists @ H_image

    def save(self) -> str:
        with open(self.path, "wb") as f:
            pickle.dump(self, f)
        return self.path
