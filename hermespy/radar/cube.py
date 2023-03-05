# -*- coding: utf-8 -*-
"""
==========
Radar Cube
==========

Radar cubes represent the raw image create after the base-band processing of radar samples.
"""

from __future__ import annotations
from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group

from hermespy.core import Executable, HDFSerializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarCube(HDFSerializable):
    """A representation of raw radar image samples."""

    data: np.ndarray
    angle_bins: np.ndarray
    velocity_bins: np.ndarray
    range_bins: np.ndarray

    def __init__(self, data: np.ndarray, angle_bins: np.ndarray, velocity_bins: np.ndarray, range_bins: np.ndarray) -> None:
        if data.shape[0] != len(angle_bins):
            raise ValueError("Data cube angle dimension does not match angle bins")

        if data.shape[1] != len(velocity_bins):
            raise ValueError("Data cube velocity dimension does not match velocity bins")

        if data.shape[2] != len(range_bins):
            raise ValueError("Data cube range dimension does not match range bins")

        self.data = data
        self.angle_bins = angle_bins
        self.velocity_bins = velocity_bins
        self.range_bins = range_bins

    def plot_range(self, title: Optional[str] = None, axes: plt.Axes | None = None) -> plt.Figure:
        """Visualize the cube's range data.

        Args:

            title (str, optional):
                Plot title.

        Returns:
            plt.Figure:
        """

        title = "Radar Range Profile" if title is None else title

        # Collapse the cube into the range-dimension
        range_profile = np.sum(self.data, axis=(0, 1), keepdims=False)

        if axes is None:
            with Executable.style_context():
                figure, axes = plt.subplots()
                figure.suptitle(title)

        else:
            figure = axes.figure

        axes.set_xlabel("Range [m]")
        axes.set_ylabel("Power")
        axes.plot(self.range_bins, range_profile)

        return figure

    def plot_range_velocity(self, title: Optional[str] = None, interpolate: bool = True) -> plt.Figure:
        """Visualize the cube's range-velocity profile.

        Args:

            title (str, optional):
                Plot title.

            interpolate (bool, optional):
                Interpolate the axis for a square profile plot.
                Enabled by default.

        Returns:
            plt.Figure:
        """

        title = "Radar Range-Doppler Profile" if title is None else title

        # Collapse the cube into the range-dimension
        range_velocity_profile = np.sum(self.data, axis=0, keepdims=False)

        figure, axes = plt.subplots()
        figure.suptitle(title)

        with Executable.style_context():
            axes.set_xlabel("Range [m]")
            axes.set_ylabel("Doppler [Hz]")

            plt.pcolormesh(self.range_bins, self.velocity_bins, range_velocity_profile, shading="auto")

            return figure

    def normalize_power(self) -> None:
        """Normalize the represented power indicators to unit maximum."""

        self.data /= self.data.max()

    @classmethod
    def from_HDF(cls: Type[RadarCube], group: Group) -> RadarCube:
        data = np.array(group["data"])
        angle_bins = np.array(group["angle_bins"], dtype=np.float_)
        velocity_bins = np.array(group["velocity_bins"], dtype=np.float_)
        range_bins = np.array(group["range_bins"], dtype=np.float_)

        return cls(data=data, angle_bins=angle_bins, velocity_bins=velocity_bins, range_bins=range_bins)

    def to_HDF(self, group: Group) -> None:
        self._write_dataset(group, "data", self.data)
        self._write_dataset(group, "angle_bins", self.angle_bins)
        self._write_dataset(group, "velocity_bins", self.velocity_bins)
        self._write_dataset(group, "range_bins", self.range_bins)
