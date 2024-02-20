# -*- coding: utf-8 -*-
"""
=============
Visualizers
=============
"""

from __future__ import annotations
from abc import ABC
from typing import Generic, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Evaluator, Signal, VAT, VT, PlotVisualization, ScatterVisualization
from hermespy.modem import ReceivingModem
from hermespy.radar import Radar, RadarCube
from .hardware_loop import HardwareLoopPlot, HardwareLoopSample
from .physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SignalPlot(HardwareLoopPlot[PlotVisualization], ABC):
    """Base class of hardware loop plots visualizing signal models."""

    __space: Literal["time", "frequency", "both"]

    def __init__(
        self, title: str = "", space: Literal["time", "frequency", "both"] = "time"
    ) -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__space = space

    @property
    def space(self) -> Literal["time", "frequency", "both"]:
        """Space in which the signal is to be plotted."""

        return self.__space

    def _initialize_subplots(self, num_streams: int) -> Tuple[plt.Figure, VAT]:
        """Initialize the suplots for a signal model visualization.

        Subroutine of :meth:`HardwareLoopPlot._prepare_plot` for classes inheriting
        from :class:`SignalPlot`.

        Args:

            num_streams (int): Then number of expected streams.

        Returns: The figure and axes of the subplots.
        """

        figure, axes = plt.subplots(num_streams, 2 if self.space == "both" else 1, squeeze=False)
        return figure, axes

    def _plot_initial_signal(self, signal: Signal, axes: VAT) -> PlotVisualization:
        # Plot transmitted signal
        return signal.plot.visualize(axes, space=self.__space)

    def _update_signal_plot(self, signal: Signal, visualization: PlotVisualization) -> None:
        """Update the plot of a signal model.

        Subroutine of :meth:`HardwareLoopPlot._update_plot` for classes inheriting
        from :class:`SignalPlot`.

        Args:

            signal (Signal): Signal model to be plotted.

            visualization (PlotVisualization): Visualization to be updated.
        """

        signal.plot.update_visualization(visualization, space=self.__space)


class HardwareLoopDevicePlot(Generic[VT], HardwareLoopPlot[VT], ABC):
    """Base class for plots of information generated by a device."""

    __device: PhysicalDevice

    def __init__(self, device: PhysicalDevice, title: str | None = None) -> None:
        """
        Args:

            device (PhysicalDevice):
                Physical device of which information is to be plotted.

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.
        """

        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__device = device

    @property
    def device(self) -> PhysicalDevice:
        """Physical device of which information is to be plotted."""

        return self.__device


class DeviceTransmissionPlot(HardwareLoopDevicePlot[PlotVisualization], SignalPlot):
    """Plot base-band signals transmitted by a device."""

    def __init__(
        self,
        device: PhysicalDevice,
        title: str | None = None,
        space: Literal["time", "frequency", "both"] = "time",
    ) -> None:
        """
        Args:

            device (PhysicalDevice):
                Physical device of which information is to be plotted.

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.

            space (Literal["time", "frequency", "both"], optional):
                Space in which the signal is to be plotted.
                By the default, the signal is plotted in the time domain.
        """

        # Initialize base classes
        HardwareLoopDevicePlot.__init__(self, device, title)
        SignalPlot.__init__(self, title, space)

    @property
    def _default_title(self) -> str:
        return "Device Transmission"

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = self._initialize_subplots(self.device.antennas.num_transmit_ports)
        return figure, axes

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> PlotVisualization:
        return self._plot_initial_signal(
            sample.drop.device_transmissions[
                self.hardware_loop.device_index(self.device)
            ].mixed_signal,
            axes,
        )

    def _update_plot(self, sample: HardwareLoopSample, visualization: PlotVisualization) -> None:
        self._update_signal_plot(
            sample.drop.device_transmissions[
                self.hardware_loop.device_index(self.device)
            ].mixed_signal,
            visualization,
        )


class DeviceReceptionPlot(HardwareLoopDevicePlot[PlotVisualization], SignalPlot):
    """Plot base-band signals received by a device."""

    def __init__(
        self,
        device: PhysicalDevice,
        title: str | None = None,
        space: Literal["time", "frequency"] = "time",
    ) -> None:
        """
        Args:

            device (PhysicalDevice):
                Physical device of which information is to be plotted.

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.

            space (Literal["time", "frequency", "both"], optional):
                Space in which the signal is to be plotted.
                By the default, the signal is plotted in the time domain.
        """

        # Initialize base classes
        HardwareLoopDevicePlot.__init__(self, device, title)
        SignalPlot.__init__(self, title, space)

    @property
    def _default_title(self) -> str:
        return "Device Reception"

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = self._initialize_subplots(self.device.antennas.num_receive_ports)
        return figure, axes

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> PlotVisualization:
        return self._plot_initial_signal(
            sample.drop.device_receptions[
                self.hardware_loop.device_index(self.device)
            ].impinging_signals[0],
            axes,
        )

    def _update_plot(self, sample: HardwareLoopSample, visualization: PlotVisualization) -> None:
        self._update_signal_plot(
            sample.drop.device_receptions[
                self.hardware_loop.device_index(self.device)
            ].impinging_signals[0],
            visualization,
        )


class EyePlot(HardwareLoopPlot[PlotVisualization]):
    """Plot eye diagrams of a received signal."""

    __modem: ReceivingModem

    def __init__(self, modem: ReceivingModem, title: str | None = None) -> None:
        """
        Args:

            modem (ReceivingModem):
                Modem of which information is to be plotted.

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.
        """

        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__modem = modem

    @property
    def _default_title(self) -> str:
        return "Eye Plot"

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        return figure, axes

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> PlotVisualization:
        if self.__modem.reception is None or self.__modem.reception.num_frames < 1:
            raise RuntimeError("No frames received yet")

        return self.__modem.reception.frames[0].signal.eye.visualize(
            ymbol_duration=self.__modem.symbol_duration, axes=axes
        )

    def _update_plot(self, sample: HardwareLoopSample, visualization: PlotVisualization) -> None:
        self.__modem.reception.frames[0].signal.eye.update_visualization(
            symbol_duration=self.__modem.symbol_duration, visualization=visualization
        )


class ReceivedConstellationPlot(HardwareLoopPlot[ScatterVisualization]):
    """Plot the constellation diagram of a received signal."""

    __modem: ReceivingModem

    def __init__(self, modem: ReceivingModem, title: str | None = None) -> None:
        """
        Args:

            modem (ReceivingModem):
                Modem of which information is to be plotted.

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.
        """

        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__modem = modem

    @property
    def _default_title(self) -> str:
        return "Received Symbol Constellation"

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        return figure, axes

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> ScatterVisualization:
        return self.__modem.reception.equalized_symbols.plot_constellation.visualize(axes=axes)

    def _update_plot(self, sample: HardwareLoopSample, visualization: ScatterVisualization) -> None:
        self.__modem.reception.equalized_symbols.plot_constellation.update_visualization(
            visualization
        )


class RadarRangePlot(HardwareLoopPlot[PlotVisualization]):
    """Plot of a radar's range-power profile.""" ""

    __radar: Radar

    def __init__(self, radar: Radar, title: str | None = None) -> None:
        """
        Args:

            radar (Radar):
                Radar of which information is to be plotted.

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.
        """

        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__radar = radar

    @property
    def _default_title(self) -> str:
        return "Range-Power Profile"

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        return figure, axes

    def __get_cube(self) -> RadarCube:
        """Fetch the radar cube from the radar reception.

        Returns: The cube.
        """

        if not self.__radar.reception:
            raise RuntimeError("Radar reception is not available")

        cube = self.__radar.reception.cube
        cube.normalize_power()  # This might be a problem since the normalization is in-place

        return self.__radar.reception.cube

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> PlotVisualization:
        cube = self.__get_cube()
        plot = cube.plot_range(axes=axes)
        # axes[0, 0].set_ylim((0.0, 1.1))
        return plot

    def _update_plot(self, sample: HardwareLoopSample, visualization: PlotVisualization) -> None:
        self.__get_cube().update_range_plot(visualization)


class HardwareLoopEvaluatorPlot(Generic[VT], HardwareLoopPlot[VT], ABC):
    """Common base class for plots of information generated by an evaluator."""

    __evaluator: Evaluator

    def __init__(self, evaluator: Evaluator, title: str = "") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__evaluator = evaluator

    @property
    def _default_title(self) -> str:
        return self.evaluator.title

    @property
    def evaluator(self) -> Evaluator:
        """Evaluator of which information is to be plotted."""

        return self.__evaluator

    @property
    def evaluator_index(self) -> int:
        """Index of the evaluator within the HardwareLoop's configuration."""

        return self.hardware_loop.evaluator_index(self.evaluator)

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        return figure, axes


class EvaluationPlot(Generic[VT], HardwareLoopEvaluatorPlot[VT]):
    """Plot of an evaluation."""

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> VT:
        return sample.evaluations[self.evaluator_index].visualize(axes)

    def _update_plot(self, sample: HardwareLoopSample, visualization: VT) -> None:
        sample.evaluations[self.evaluator_index].update_visualization(visualization)


class ArtifactPlot(HardwareLoopEvaluatorPlot[PlotVisualization]):
    """Plot of an evaluation's scalar artifact."""

    __artifact_queue: np.ndarray

    def __init__(
        self, evaluator: Evaluator, title: str | None = None, queue_length: int = 20
    ) -> None:
        # Initialize base class
        HardwareLoopEvaluatorPlot.__init__(self, evaluator, title)

        # Initialize class attributes
        self.__artifact_queue = np.nan * np.ones((queue_length, 1), dtype=float)
        self.__artifact_indices = np.arange(queue_length)

    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = HardwareLoopEvaluatorPlot._prepare_plot(self)

        ax: plt.Axes = axes.flat[0]
        ax.set_xlabel("Drop Index")
        ax.set_ylabel(self.evaluator.abbreviation)

        return figure, axes

    def __update_artifact_queue(self, sample: HardwareLoopSample) -> None:
        """Update the artifact queue with the most recent artifact.

        Subroutine of :meth:`ArtifactPlot._update_plot` and :meth:`ArtifactPlot._initial_plot`.

        Args:

            sample (HardwareLoopSample): The most recent sample.
        """

        self.__artifact_queue = np.roll(self.__artifact_queue, 1)
        self.__artifact_queue[0] = sample.artifacts[
            self.hardware_loop.evaluator_index(self.evaluator)
        ].to_scalar()

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> PlotVisualization:
        # Update artifact queue
        self.__update_artifact_queue(sample)

        # Plot artifact queue
        lines = np.empty_like(axes, dtype=np.object_)
        lines[0, 0] = axes[0, 0].plot(self.__artifact_indices, self.__artifact_queue)

        return PlotVisualization(axes[0, 0].get_figure(), axes, lines)

    def _update_plot(self, sample: HardwareLoopSample, visualization: PlotVisualization) -> None:
        # Update artifact queue
        self.__update_artifact_queue(sample)

        # Update visualization
        visualization.lines[0, 0][0].set_ydata(self.__artifact_queue)
