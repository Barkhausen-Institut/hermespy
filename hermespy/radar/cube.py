# -*- coding: utf-8 -*-
"""
==========
Radar Cube
==========

Radar cubes represent the raw image create after the base-band processing of radar samples.
"""

from __future__ import annotations
from typing import Literal, Type

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

    __data: np.ndarray
    __angle_bins: np.ndarray
    __velocity_bins: np.ndarray
    __range_bins: np.ndarray

    def __init__(self, data: np.ndarray, angle_bins: np.ndarray | None = None, velocity_bins: np.ndarray | None = None, range_bins: np.ndarray | None = None) -> None:
        """
        Args:

            data (np.ndarray):
                Raw radar cube data.
                Three-dimensional real-valued numpy tensor :math:`\\mathbb{R}^{A \\times B \\times C}`, where
                :math:`A` denotes the number of discrete angle of arrival bins,
                :math:`B` denotes the number of discrete doppler frequency bins,
                and :math:`C` denotes the number of discrete range bins.

            angle_bins (np.ndarray):
                Numpy matrix specifying the represented discrete angle of arrival bins.
                Must be of dimension :math:`\\mathbb{R}^{A \\times 2}`,
                the second dimension denoting azimuth and zenith of arrival in radians, respectively.

            velocity_bins (np.ndarray):
                Numpy vector specifying the represented discrete doppler frequency bins in Hz.
                Must be of dimension :math:`\\mathbb{R}^{B}`.

            range_bins (np.ndarray):
                Numpy vector specifying the represented discrete range bins in :math:`\\mathrm{m}`.
                Must be of dimension :math:`\\mathbb{R}^{C}`.

        Raises:

            ValueError:
                If the argument numpy arrays have unexpected dimensions or if their dimensions don't match.
        """

        if data.ndim != 3:
            raise ValueError(f"Cube data must be a three-dimensional numpy tensor (has {data.ndim} dimenions)")

        # Infer angle bins
        if angle_bins is None:
            if data.shape[0] == 1:
                angle_bins = np.array([[0, 0]], dtype=np.float_)

            else:
                raise ValueError("Can't infer angle bins from data cube")

        # Infer velocity bins
        if velocity_bins is None:
            if data.shape[0] == 1:
                velocity_bins = np.array([0], dtype=np.float_)

            else:
                raise ValueError("Can't infer velocity bins from data cube")

        # Infer range bins
        range_bins = np.arange(data.shape[2], dtype=np.float_) if range_bins is None else range_bins

        if data.shape[0] != len(angle_bins):
            raise ValueError("Data cube angle dimension does not match angle bins")

        if data.shape[1] != len(velocity_bins):
            raise ValueError("Data cube velocity dimension does not match velocity bins")

        if data.shape[2] != len(range_bins):
            raise ValueError("Data cube range dimension does not match range bins")

        self.__data = data
        self.__angle_bins = angle_bins
        self.__velocity_bins = velocity_bins
        self.__range_bins = range_bins

    @property
    def data(self) -> np.ndarray:
        """Raw radar cube data.

        Three-dimensional real-valued numpy tensor :math:`\\mathbb{R}^{A \\times B \\times C}`, where
        :math:`A` denotes the number of discrete angle of arrival bins,
        :math:`B` denotes the number of discrete doppler frequency bins,
        and :math:`C` denotes the number of discrete range bins.

        Returns: Radar cube numpy tensor.
        """

        return self.__data

    @property
    def angle_bins(self) -> np.ndarray:
        """Discrete angle estimation bins.

        Returns:
            Numpy matrix of dimension :math:`\\mathbb{R}^{A \\times 2}`,
            the second dimension denoting azimuth and zenith of arrival in radians, respectively.
        """

        return self.__angle_bins

    @property
    def velocity_bins(self) -> np.ndarray:
        """Discrete velocity estimation bins.


        Returns:
            Numpy vector specifying the represented discrete doppler frequency bins in Hz.
        """

        return self.__velocity_bins

    @property
    def range_bins(self) -> np.ndarray:
        """Discrete range estimation bins.


        Returns:
                Numpy vector specifying the represented discrete range bins in :math:`\\mathrm{m}`.
        """

        return self.__range_bins

    def plot_range(self, title: str | None = None, axes: plt.Axes | None = None, scale: Literal["lin", "log"] = "lin") -> plt.Figure:
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

        if scale == "lin":
            axes.plot(self.range_bins, range_profile)

        elif scale == "log":
            axes.semilogy(self.range_bins, range_profile)

        else:
            raise ValueError(f"Unsupported plotting scale option '{scale}'")

        return figure

    def plot_range_velocity(self, title: str | None = None, interpolate: bool = True) -> plt.Figure:
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

        with Executable.style_context():
            figure, axes = plt.subplots()
            figure.suptitle(title)

            axes.set_xlabel("Range [m]")
            axes.set_ylabel("Doppler [Hz]")

            plt.pcolormesh(self.range_bins, self.velocity_bins, range_velocity_profile, shading="auto")

            return figure

    def normalize_power(self) -> None:
        """Normalize the represented power indicators to unit maximum."""

        self.__data /= self.__data.max()

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
