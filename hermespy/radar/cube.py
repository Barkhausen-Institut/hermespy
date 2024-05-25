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
from scipy.interpolate import bisplrep, bisplev

from hermespy.core import (
    HDFSerializable,
    PlotVisualization,
    QuadMeshVisualization,
    VisualizableAttribute,
    VAT,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _RangePlot(VisualizableAttribute[PlotVisualization]):
    """Visualizable attribute for plotting a range-power profiles."""

    __cube: RadarCube

    def __init__(self, cube: RadarCube) -> None:
        self.__cube = cube

    @property
    def title(self) -> str:
        return "Radar Range-Power Profile"

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, scale: Literal["lin", "log"] = "lin", **kwargs
    ) -> PlotVisualization:

        _ax: plt.Axes = axes[0, 0]
        _ax.set_xlabel("Range [m]")
        _ax.set_ylabel("Power")

        range_profile = np.sum(self.__cube.data, axis=(0, 1), keepdims=False)
        lines = np.empty((1, 1), dtype=np.object_)
        if scale == "lin":
            lines[0, 0] = _ax.plot(self.__cube.range_bins, range_profile)

        elif scale == "log":
            lines[0, 0] = _ax.semilogy(self.__cube.range_bins, range_profile)

        return PlotVisualization(figure, axes, lines)

    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        range_profile = np.sum(self.__cube.data, axis=(0, 1), keepdims=False)
        visualization.lines[0, 0][0].set_ydata(range_profile)


class _RangeVelocityPlot(VisualizableAttribute[QuadMeshVisualization]):
    """Visualizable attribute for plotting range-velocity profiles."""

    __cube: RadarCube

    def __init__(self, cube: RadarCube) -> None:
        self.__cube = cube

    @property
    def title(self) -> str:
        return "Radar Range-Doppler Profile"

    def _prepare_visualization(
        self,
        figure: plt.Figure | None,
        axes: VAT,
        scale: Literal["frequency", "velocity"] | None = None,
        **kwargs,
    ) -> QuadMeshVisualization:

        _ax: plt.Axes = axes[0, 0]

        if scale is None:
            _scale = "velocity" if self.__cube.carrier_frequency > 0 else "frequency"
        else:
            _scale = scale

        # Compute velocity axis bins depending on the scale
        velocity_axis = (
            self.__cube.velocity_bins if _scale == "velocity" else self.__cube.doppler_bins
        )

        _ax.set_xlabel("Range [m]")
        _ax.set_ylabel("Doppler [Hz]" if _scale == "frequency" else "Velocity [m/s]")

        range_velocity_profile = np.abs(np.sum(self.__cube.data, axis=0, keepdims=False))
        range_velocity_profile -= np.min(range_velocity_profile)
        range_velocity_profile /= np.max(range_velocity_profile)

        mesh = _ax.pcolormesh(
            self.__cube.range_bins, velocity_axis, range_velocity_profile, shading="auto"
        )
        return QuadMeshVisualization(figure, axes, mesh)

    def _update_visualization(self, visualization: QuadMeshVisualization, **kwargs) -> None:
        range_velocity_profile = np.abs(np.sum(self.__cube.data, axis=0, keepdims=False))
        range_velocity_profile -= np.min(range_velocity_profile)
        range_velocity_profile /= np.max(range_velocity_profile)
        visualization.mesh.set_array(range_velocity_profile)


class _AnglePlot(VisualizableAttribute[QuadMeshVisualization]):

    __cube: RadarCube

    def __init__(self, cube: RadarCube) -> None:
        self.__cube = cube

    @property
    def title(self) -> str:
        return "Angle Power Profile"

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> QuadMeshVisualization:
        _ax: plt.Axes = axes[0, 0]
        _ax.grid(True)
        _ax.set_xlabel("Azimuth")
        _ax.set_ylabel("Zenith")

        default_data = np.zeros((91, 360))
        default_data[0, 0] = 1
        mesh = _ax.pcolormesh(
            np.arange(360), np.arange(91), default_data, shading="auto", norm="linear", linewidth=0
        )
        return QuadMeshVisualization(figure, axes, mesh)

    def _update_visualization(self, visualization: QuadMeshVisualization, **kwargs) -> None:

        angles = 180 * self.__cube.angle_bins / np.pi
        angle_profile = np.sum(self.__cube.data, axis=(1, 2), keepdims=False)

        splines = bisplrep(angles[:, 0], angles[:, 1], angle_profile, kx=1, ky=1, quiet=True)
        interpolated_image = bisplev(np.arange(360), np.arange(91), splines)

        interpolated_image -= interpolated_image.min()
        if interpolated_image.max() > 0:
            interpolated_image /= interpolated_image.max()

        visualization.mesh.set_array(interpolated_image.T)


class RadarCube(HDFSerializable):
    """A representation of raw radar image samples."""

    __data: np.ndarray
    __angle_bins: np.ndarray
    __doppler_bins: np.ndarray
    __range_bins: np.ndarray
    __carrier_frequency: float

    __range_plot: _RangePlot
    __range_velocity_plot: _RangeVelocityPlot
    __angle_plot: _AnglePlot

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

        self.__range_plot = _RangePlot(self)
        self.__range_velocity_plot = _RangeVelocityPlot(self)
        self.__angle_plot = _AnglePlot(self)

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

    @property
    def carrier_frequency(self) -> float:
        """Central carrier frequency of the radar in Hz."""

        return self.__carrier_frequency

    @property
    def plot_range(self) -> _RangePlot:
        """Visualize the cube's range-power profile.

        Args:

            title (str, optional):
                Plot title.

            scale (Literal['lin', 'log'], optional):
                Plot the power axis in linear or logarithmic scale.
                If not specified, linear scaling is preferred.

        Returns: The generated line plot.
        """

        return self.__range_plot

    @property
    def plot_range_velocity(self) -> _RangeVelocityPlot:
        """Visualize the cube's range-velocity profile.

        Args:

            title (str, optional):
                Plot title.

            scale (Literal['frequency', 'velocity'], optional):
                Plot the velocity axis in frequency (Hz) or velocity units (m/s).
                If not specified, plotting in velocity is preferred, if the carrier frequency is known.

        Returns: The generated image plot.
        """

        return self.__range_velocity_plot

    @property
    def plot_angles(self) -> _AnglePlot:
        """Visualize the cube's angle-power profile.

        Args:

            title (str, optional):
                Plot title.

        Returns: The generated image plot.
        """

        return self.__angle_plot

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
