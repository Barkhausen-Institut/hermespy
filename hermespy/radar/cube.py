# -*- coding: utf-8 -*-
"""
==========
Radar Cube
==========

Radar cubes represent the raw image create after the base-band processing of radar samples.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarCube(object):
    """A representation of raw radar image samples."""

    data: np.ndarray
    angle_bins: np.ndarray
    velocity_bins: np.ndarray
    range_bins: np.ndarray

    def __init__(self,
                 data: np.ndarray,
                 angle_bins: np.ndarray,
                 velocity_bins: np.ndarray,
                 range_bins: np.ndarray) -> None:
        
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

    def plot_range(self,
                   title: Optional[str] = None) -> plt.Figure:
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

        figure, axes = plt.subplots()
        figure.suptitle(title)

        axes.set_xlabel("Range [m]")
        axes.set_ylabel("Power")
        axes.plot(self.range_bins, range_profile)

        return figure

    def plot_range_velocity(self,
                            title: Optional[str] = None,
                            interpolate: bool = True) -> plt.Figure:
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

        title = "Radar Range-Velocity Profile" if title is None else title

        # Collapse the cube into the range-dimension
        range_velocity_profile = np.sum(self.data, axis=0, keepdims=False)

        figure, axes = plt.subplots()
        figure.suptitle(title)

        axes.set_xlabel("Range [m]")
        axes.set_ylabel("Velocity [m/s]")
        axes.imshow(range_velocity_profile, aspect='auto')

        return figure

    def normalize_power(self) -> None:
        """Normalize bin powers."""
        
        self.data /= abs(self.data).max()
