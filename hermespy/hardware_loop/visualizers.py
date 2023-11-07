# -*- coding: utf-8 -*-
"""
=============
Visualizers
=============
"""

from __future__ import annotations
from abc import ABC
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Evaluator, Signal, VAT
from hermespy.modem import ReceivingModem
from hermespy.radar import Radar
from .hardware_loop import HardwareLoopPlot, HardwareLoopSample
from .physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SignalPlot(HardwareLoopPlot, ABC):
    __space: Literal["time", "frequency"]

    def __init__(self, title: str = "", space: Literal["time", "frequency"] = "time") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__space = space

    def _plot_signal(self, signal: Signal) -> None:
        # Clear plot
        self.axes[0, 0].clear()

        # Plot transmitted signal
        signal.plot(axes=self.axes, space=self.__space)

        # Update plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class HardwareLoopDevicePlot(HardwareLoopPlot, ABC):
    __device: PhysicalDevice

    def __init__(self, device: PhysicalDevice, title: str = "") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__device = device

    @property
    def device(self) -> PhysicalDevice:
        """Physical device of which information is to be plotted."""

        return self.__device


class DeviceTransmissionPlot(HardwareLoopDevicePlot, SignalPlot):
    """Plot base-band signals transmitted by a device."""

    def __init__(self, device: PhysicalDevice, title: str = "", space: Literal["time", "frequency"] = "time") -> None:
        # Initialize base classes
        HardwareLoopDevicePlot.__init__(self, device, title)
        SignalPlot.__init__(self, title, space)

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(self.device.antennas.num_transmit_antennas, 1)
        figure.suptitle(self.title if self.title != "" else "Transmitted Signal")

        return figure, axes

    def _update_plot(self, sample: HardwareLoopSample) -> None:
        self._plot_signal(sample.drop.device_transmissions[self.hardware_loop.device_index(self.device)].mixed_signal)


class DeviceReceptionPlot(HardwareLoopDevicePlot, SignalPlot):
    """Plot base-band signals received by a device."""

    def __init__(self, device: PhysicalDevice, title: str = "", space: Literal["time", "frequency"] = "time") -> None:
        # Initialize base classes
        HardwareLoopDevicePlot.__init__(self, device, title)
        SignalPlot.__init__(self, title, space)

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(self.device.antennas.num_transmit_antennas, 1, squeeze=False)
        figure.suptitle(self.title if self.title != "" else "Transmitted Signal")

        return figure, axes

    def _update_plot(self, sample: HardwareLoopSample) -> None:
        self._plot_signal(sample.drop.device_receptions[self.hardware_loop.device_index(self.device)].impinging_signals[0])


class EyePlot(HardwareLoopPlot):
    __modem: ReceivingModem

    def __init__(self, modem: ReceivingModem, title: str = "") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__modem = modem

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        figure.suptitle(self.title if self.title != "" else "Eye Diagram")
        return figure, axes

    def _update_plot(self, sample: HardwareLoopSample) -> None:
        # Clear plot
        self.axes[0, 0].clear()

        # Plot eye diagram
        symbol_duration = self.__modem.symbol_duration
        if self.__modem.reception.num_frames > 0:
            self.__modem.reception.frames[0].signal.plot_eye(symbol_duration=symbol_duration, axes=self.axes)
        else:
            self.__modem.reception.signal.plot_eye(symbol_duration=symbol_duration, axes=self.axes)

        # Update plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class ReceivedConstellationPlot(HardwareLoopPlot):
    __modem: ReceivingModem

    def __init__(self, modem: ReceivingModem, title: str = "") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__modem = modem

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        figure.suptitle(self.title if self.title != "" else "Eye Diagram")

        return figure, axes

    def _update_plot(self, sample: HardwareLoopSample) -> None:
        ax: plt.Axes = self.axes.flat[0]

        # Clear plot
        ax.clear()

        # Plot constellation diagram
        self.__modem.reception.equalized_symbols.plot_constellation(axes=ax)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

        # Update plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class RadarRangePlot(HardwareLoopPlot):
    __radar: Radar

    def __init__(self, radar: Radar, title: str = "") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__radar = radar

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        figure.suptitle(self.title if self.title != "" else "Eye Diagram")
        return figure, axes

    def _update_plot(self, sample: HardwareLoopSample) -> None:
        ax: plt.Axes = self.axes.flat[0]

        # Clear plot
        ax.clear()

        # Plot constellation diagram
        cube = self.__radar.reception.cube
        cube.normalize_power()

        cube.plot_range(axes=ax)
        ax.set_ylim((0.0, 1.0))

        # Update plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class HardwareLoopEvaluatorPlot(HardwareLoopPlot, ABC):
    __evaluator: Evaluator

    def __init__(self, evaluator: Evaluator, title: str = "") -> None:
        # Initialize base class
        HardwareLoopPlot.__init__(self, title)

        # Initialize class attributes
        self.__evaluator = evaluator

    @property
    def evaluator(self) -> Evaluator:
        """Evaluator of which information is to be plotted."""

        return self.__evaluator

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False)
        figure.suptitle(self.title if self.title != "" else self.evaluator.title)
        return figure, axes


class EvaluationPlot(HardwareLoopEvaluatorPlot):
    def _update_plot(self, sample: HardwareLoopSample) -> None:
        ax: plt.Axes = self.axes.flat[0]

        # Clear plot
        ax.clear()

        # Plot constellation diagram
        evaluator_index = self.hardware_loop.evaluator_index(self.evaluator)
        sample.evaluations[evaluator_index].plot(axes=self.axes)

        # Update plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class ArtifactPlot(HardwareLoopEvaluatorPlot):
    __artifact_queue: np.ndarray

    def __init__(self, evaluator: Evaluator, title: str = "", queue_length: int = 20) -> None:
        # Initialize base class
        HardwareLoopEvaluatorPlot.__init__(self, evaluator, title)

        # Initialize class attributes
        self.__artifact_queue = np.nan * np.ones((queue_length, 1), dtype=float)

    def _prepare_figure(self) -> Tuple[plt.Figure, VAT]:
        figure, axes = HardwareLoopEvaluatorPlot._prepare_figure(self)

        ax: plt.Axes = axes.flat[0]
        ax.set_xlabel("Drop Index")
        ax.set_ylabel(self.evaluator.abbreviation)

        return figure, axes

    def _update_plot(self, sample: HardwareLoopSample) -> None:
        ax: plt.Axes = self.axes.flat[0]

        # Clear plot
        ax.clear()

        # Update artifact queue
        self.__artifact_queue = np.roll(self.__artifact_queue, 1)
        self.__artifact_queue[0] = sample.artifacts[self.hardware_loop.evaluator_index(self.evaluator)].to_scalar()

        # Update plot
        ax.plot(np.arange(len(self.__artifact_queue)), self.__artifact_queue)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
