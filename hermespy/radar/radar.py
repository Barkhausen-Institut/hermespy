# -*- coding: utf-8 -*-
"""
======================
Radar Device Operation
======================


.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

       subgraph Radar

           direction LR

           subgraph Waveform
               Modulation
               TargetEstimation --- Demodulation
           end

           subgraph BeamForming

               TxBeamform[Tx Beamforming]
               RxBeamform[Rx Beamforming]
           end

           Modulation --> TxBeamform
           Demodulation --- RxBeamform

       end

       subgraph Device

           direction TB
           txslot>Tx Slot]
           rxslot>Rx Slot]
       end

   estimations{{Target Estimations}}
   txsignal{{Tx Signal Model}}
   rxsignal{{Rx Signal Model}}

   TxBeamform --> txsignal
   RxBeamform --- rxsignal
   txsignal --> txslot
   rxsignal --- rxslot

   TargetEstimation --- estimations
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core.signal_model import Signal
from ..core.device import DuplexOperator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PointDetection(object):
    """A single radar point detection."""

    __position: np.ndarray      # Cartesian position of the detection in m
    __velocity: np.ndarray      # Velocity of the detection in m/s
    __power: float              # Power of the detection

    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 power: float) -> None:
        """
        Args:

            position (np.ndarray):
                Cartesian position of the detection in cartesian coordinates.

            velocity (np.ndarray):
                Velocity vector of the detection in m/s

            power (float):
                Power of the detection.

        Raises:
            ValueError:
                If `position` is not three-dimensional.
                If `velocity` is not three-dimensional.
                If `power` is smaller or equal to zero.
        """

        if position.ndim != 3:
            raise ValueError("Position must be a three-dimensional vector.")

        if velocity.ndim != 3:
            raise ValueError("Velocity must be a three-dimensional vector.")

        if power <= 0.:
            raise ValueError("Detected power must be greater than zero")

        self.__position = position
        self.__velocity = velocity
        self.__power = power

    @property
    def position(self) -> np.ndarray:
        """Position of the detection.

        Returns:
            np.ndarray: Cartesian position in m.
        """

        return self.__position

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the detection.

        Returns:
            np.ndarray: Velocity vector in m/s.
        """

        return self.__velocity

    @property
    def power(self) -> float:
        """Detected power.

        Returns:
            float: Power.
        """

        return self.__power


class RadarCube(object):

    data: np.ndarray
    angle_bins: np.ndarray
    velocity_bins: np.ndarray
    range_bins: np.ndarray

    def __init__(self,
                 data: np.ndarray,
                 angle_bins: np.ndarray,
                 velocity_bins: np.ndarray,
                 range_bins: np.ndarray) -> None:

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


class RadarWaveform(object):
    """Base class for waveform generation of radars."""

    @abstractmethod
    def ping(self) -> Signal:
        """Generate a single radar frame.

        Returns:
            Signal: Model of the radar frame.
        """
        ...

    @abstractmethod
    def estimate(self,
                 signal: Signal) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """The optional sampling rate required to process this waveform.

        Returns:
            sampling_rate (float): Sampling rate in Hz.
        """
        ...

    @property
    @abstractmethod
    def range_bins(self) -> np.ndarray:
        """Sample bins of the depth sensing.

        Returns:
            np.ndarray: Ranges in m.
        """
        ...

    @property
    @abstractmethod
    def velocity_bins(self) -> np.ndarray:
        """Sample bins of the radial velocity sensing.

        Returns:
            np.ndarray: Velocities in m/s.
        """
        ...


class Radar(DuplexOperator):
    """HermesPy representation of a mono-static radar sensing its environment."""

    waveform: Optional[RadarWaveform]

    def __init__(self) -> None:

        self.waveform = None

        DuplexOperator.__init__(self)

    def transmit(self, duration: float = 0.) -> Tuple[Signal]:

        if not self.waveform:
            raise RuntimeError("Radar waveform not specified")

        # Generate the radar waveform
        transmitted_signal = self.waveform.ping()

        # Transmit signal over the occupied device slot (if the radar is attached to a device)
        if self._transmitter.attached:
            self._transmitter.slot.add_transmission(self._transmitter, transmitted_signal)

        return transmitted_signal,

    def receive(self) -> Tuple[RadarCube]:

        # Retrieve signal from receiver slot
        signal = self._receiver.signal.resample(self.waveform.sampling_rate)

        # Build the radar cube by generating a beam-forming line over all angles of interest
        angles_of_interest = np.array([[0., 0.]], dtype=float)

        range_bins = self.waveform.range_bins
        velocity_bins = self.waveform.velocity_bins

        cube_data = np.empty((len(angles_of_interest),
                              len(velocity_bins),
                              len(range_bins)), dtype=float)

        for aoi_idx, aoi in enumerate(angles_of_interest):

            # ToDo: Beamforming

            # Process the single angular line by the waveform generator
            line = self.waveform.estimate(signal)

            cube_data[aoi_idx, ::] = line

        # Create radar cube object
        cube = RadarCube(cube_data, angles_of_interest, velocity_bins, range_bins)

        return cube,

    @property
    def sampling_rate(self) -> float:

        return self.waveform.sampling_rate

    @property
    def frame_duration(self) -> float:
        pass

    @property
    def energy(self) -> float:

        return 1.0  # ToDo: Implement
