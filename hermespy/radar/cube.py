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
from scipy.constants import speed_of_light

from hermespy.core import Executable, HDFSerializable, PlotVisualization, VAT

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarCube(HDFSerializable):
    """A representation of raw radar image samples."""

    __data: np.ndarray
    __angle_bins: np.ndarray
    __doppler_bins: np.ndarray
    __range_bins: np.ndarray
    __carrier_frequency: float

    def __init__(
        self,
        data: np.ndarray,
        angle_bins: np.ndarray | None = None,
        doppler_bins: np.ndarray | None = None,
        range_bins: np.ndarray | None = None,
        carrier_frequency: float = 0.0,
    ) -> None:
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

            doppler_bins (np.ndarray):
                Numpy vector specifying the represented discrete doppler frequency shift bins in Hz.
                Must be of dimension :math:`\\mathbb{R}^{B}`.

            range_bins (np.ndarray):
                Numpy vector specifying the represented discrete range bins in :math:`\\mathrm{m}`.
                Must be of dimension :math:`\\mathbb{R}^{C}`.

            carrier_frequency (float, optional):
                Central carrier frequency of the radar in Hz.
                Zero by default.

        Raises:

            ValueError:
                If the argument numpy arrays have unexpected dimensions or if their dimensions don't match.
        """

        if data.ndim != 3:
            raise ValueError(
                f"Cube data must be a three-dimensional numpy tensor (has {data.ndim} dimenions)"
            )

        # Infer angle bins
        if angle_bins is None:
            if data.shape[0] == 1:
                angle_bins = np.array([[0, 0]], dtype=np.float_)

            else:
                raise ValueError("Can't infer angle bins from data cube")

        # Infer velocity bins
        if doppler_bins is None:
            if data.shape[1] == 1:
                doppler_bins = np.array([0], dtype=np.float_)

            else:
                raise ValueError("Can't infer velocity bins from data cube")

        # Infer range bins
        range_bins = np.arange(data.shape[2], dtype=np.float_) if range_bins is None else range_bins

        if data.shape[0] != len(angle_bins):
            raise ValueError("Data cube angle dimension does not match angle bins")

        if data.shape[1] != len(doppler_bins):
            raise ValueError("Data cube velocity dimension does not match doppler bins")

        if data.shape[2] != len(range_bins):
            raise ValueError("Data cube range dimension does not match range bins")

        if carrier_frequency < 0:
            raise ValueError(f"Carrier frequency must be non-negtative (not {carrier_frequency})")

        self.__data = data
        self.__angle_bins = angle_bins
        self.__doppler_bins = doppler_bins
        self.__range_bins = range_bins
        self.__carrier_frequency = carrier_frequency

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
    def doppler_bins(self) -> np.ndarray:
        """Discrete doppler shift estimation bins.

        Returns:
            Numpy vector specifying the represented discrete doppler frequency bins in Hz.
        """

        return self.__doppler_bins

    @property
    def velocity_bins(self) -> np.ndarray:
        """Discrete doppler estimation bins.

        Returns:
            Numpy vector specifying the represented discrete doppler velocity bins in :math:`\\mathrm{m/s}`.

        Raises:
            RuntimeError: If the carrier frequency is not specified.
        """

        if self.__carrier_frequency <= 0.0:
            raise RuntimeError("Carrier frequency not specified")

        return self.__doppler_bins * speed_of_light / self.__carrier_frequency

    @property
    def range_bins(self) -> np.ndarray:
        """Discrete range estimation bins.


        Returns:
                Numpy vector specifying the represented discrete range bins in :math:`\\mathrm{m}`.
        """

        return self.__range_bins

    def plot_range(
        self,
        title: str | None = None,
        axes: VAT | plt.Axes | None = None,
        scale: Literal["lin", "log"] = "lin",
    ) -> PlotVisualization:
        """Visualize the cube's range data.

        Args:

            title (str, optional):
                Plot title.

        Returns: Newly generated visualization.
        """

        title = "Radar Range Profile" if title is None else title

        # Collapse the cube into the range-dimension
        range_profile = np.sum(self.data, axis=(0, 1), keepdims=False)

        figure: plt.Figure
        if axes is None:
            with Executable.style_context():
                figure, _axes = plt.subplots(1, 1, squeeze=False)
                figure.suptitle(title)

        else:
            _axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
            figure = _axes[0, 0].get_figure()

        _axes[0, 0].set_xlabel("Range [m]")
        _axes[0, 0].set_ylabel("Power")

        lines = np.empty((1, 1), dtype=np.object_)
        if scale == "lin":
            lines[0, 0] = _axes[0, 0].plot(self.range_bins, range_profile)

        elif scale == "log":
            lines[0, 0] = _axes[0, 0].semilogy(self.range_bins, range_profile)

        else:
            raise ValueError(f"Unsupported plotting scale option '{scale}'")

        return PlotVisualization(figure, _axes, lines)

    def update_range_plot(self, visualization: PlotVisualization) -> None:
        # Collapse the cube into the range-dimension
        range_profile = np.sum(self.data, axis=(0, 1), keepdims=False)
        visualization.lines[0, 0][0].set_ydata(range_profile)

    def plot_range_velocity(
        self,
        title: str | None = None,
        interpolate: bool = True,
        scale: Literal["frequency", "velocity"] | None = None,
    ) -> plt.Figure:
        """Visualize the cube's range-velocity profile.

        Args:

            title (str, optional):
                Plot title.

            interpolate (bool, optional):
                Interpolate the axis for a square profile plot.
                Enabled by default.

            scale (Literal['frequency', 'velocity'], optional):
                Plot the velocity axis in frequency (Hz) or velocity units (m/s).
                If not specified, plotting in velocity is preferred, if the carrier frequency is known.

        Returns:
            plt.Figure:
        """

        title = "Radar Range-Doppler Profile" if title is None else title

        if scale is None:
            _scale = "velocity" if self.__carrier_frequency > 0 else "frequency"
        else:
            _scale = scale

        # Compute velocity axis bins depending on the scale
        velocity_axis = self.velocity_bins if _scale == "velocity" else self.doppler_bins

        # Collapse the cube into the range-dimension
        range_velocity_profile = np.sum(self.data, axis=0, keepdims=False)

        with Executable.style_context():
            figure, axes = plt.subplots()
            figure.suptitle(title)

            axes.set_xlabel("Range [m]")
            axes.set_ylabel("Doppler [Hz]" if _scale == "frequency" else "Velocity [m/s]")

            axes.pcolormesh(self.range_bins, velocity_axis, range_velocity_profile, shading="auto")

            return figure

    def normalize_power(self) -> None:
        """Normalize the represented power indicators to unit maximum."""

        self.__data = self.__data / self.__data.max()

    @classmethod
    def from_HDF(cls: Type[RadarCube], group: Group) -> RadarCube:
        data = np.array(group["data"])
        angle_bins = np.array(group["angle_bins"], dtype=np.float_)
        doppler_bins = np.array(group["doppler_bins"], dtype=np.float_)
        range_bins = np.array(group["range_bins"], dtype=np.float_)
        carrier_frequency = group.attrs.get("carrier_frequency", 0.0)

        return cls(
            data=data,
            angle_bins=angle_bins,
            doppler_bins=doppler_bins,
            range_bins=range_bins,
            carrier_frequency=carrier_frequency,
        )

    def to_HDF(self, group: Group) -> None:
        self._write_dataset(group, "data", self.data)
        self._write_dataset(group, "angle_bins", self.angle_bins)
        self._write_dataset(group, "doppler_bins", self.doppler_bins)
        self._write_dataset(group, "range_bins", self.range_bins)
        group.attrs["carrier_frequency"] = self.__carrier_frequency
