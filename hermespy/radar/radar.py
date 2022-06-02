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

from hermespy.beamforming import ReceiveBeamformer, TransmitBeamformer
from hermespy.core import DuplexOperator, Signal


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
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

        self.position = position
        self.velocity = velocity
        self.power = power

    @property
    def position(self) -> np.ndarray:
        """Position of the detection.

        Returns:
            np.ndarray: Cartesian position in m.
            
        Raises:
            ValueError: If position is not a three-dimensional vector.
        """

        return self.__position
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        
        if value.ndim != 1 or len(value) != 3:
            raise ValueError("Position must be a three-dimensional vector")
        
        self.__position = value
        
    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the detection.

        Returns:
            np.ndarray: Velocity vector in m/s.
            
        Raises:
            ValueError: If velocity is not a three-dimensional vector.
        """

        return self.__velocity
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        
        if value.ndim != 1 or len(value) != 3:
            raise ValueError("Velocity must be a three-dimensional vector")
        
        self.__velocity = value

    @property
    def power(self) -> float:
        """Detected power.

        Returns:
            float: Power.
            
        Raises:
            ValueError: If `power` is smaller or equal to zero.
        """

        return self.__power
    
    @power.setter
    def power(self, value: float) -> None:
        
        if value <= 0.:
            raise ValueError("Detected power must be greater than zero")
        
        self.__power = value


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
                   title: Optional[str] = None,
                   axes: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Visualize the cube's range data.

        Args:

            title (str, optional):
                Plot title.
                
            axes (Optional[plt.Axes], optional):
                Matplotlib axes to plot the graph to.
                If none are provided, a new figure is created.

        Returns:
        
            Optional[plt.Figure]:
                The visualization figure.
                `None` if axes were provided.
        """

        title = "Radar Range Profile" if title is None else title
        figure: Optional[plt.Figure] = None

        # Collapse the cube into the range-dimension
        range_profile = np.sum(self.data, axis=(0, 1), keepdims=False)

        # Create a new figure if no axes were provided
        if axes is None:
            
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
        ...  # pragma: no cover

    @abstractmethod
    def estimate(self,
                 signal: Signal) -> np.ndarray:
        ...  # pragma: no cover

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """The optional sampling rate required to process this waveform.

        Returns:
            sampling_rate (float): Sampling rate in Hz.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def range_bins(self) -> np.ndarray:
        """Sample bins of the depth sensing.

        Returns:
            np.ndarray: Ranges in m.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def velocity_bins(self) -> np.ndarray:
        """Sample bins of the radial velocity sensing.

        Returns:
            np.ndarray: Velocities in m/s.
        """
        ...  # pragma: no cover


class Radar(DuplexOperator):
    """HermesPy representation of a mono-static radar sensing its environment."""

    waveform: Optional[RadarWaveform]
    __transmit_beamformer: Optional[TransmitBeamformer]
    __receive_beamformer: Optional[ReceiveBeamformer]

    def __init__(self) -> None:

        self.waveform = None
        self.receive_beamformer = None
        self.transmit_beamformer = None

        DuplexOperator.__init__(self)
        
    @property
    def transmit_beamformer(self) -> Optional[TransmitBeamformer]:
        """Beamforming applied during signal transmission.
        
        Returns:
        
            The beamformer.
            `None`, if no beamformer is configured during transmission.
        """
        
        return self.__transmit_beamformer
    
    @transmit_beamformer.setter
    def transmit_beamformer(self, value: Optional[TransmitBeamformer]) -> None:
        
        if value is None:
            
            self.__transmit_beamformer = None
            
        else:
            
            value.operator = self
            self.__transmit_beamformer = value
            
    @property
    def receive_beamformer(self) -> Optional[ReceiveBeamformer]:
        """Beamforming applied during signal transmission.
        
        Returns:
        
            The beamformer.
            `None`, if no beamformer is configured during transmission.
        """
        
        return self.__receive_beamformer
    
    @receive_beamformer.setter
    def receive_beamformer(self, value: Optional[ReceiveBeamformer]) -> None:
        
        if value is None:
            
            self.__receive_beamformer = None
            
        else:
            
            value.operator = self
            self.__receive_beamformer = value
            
    @property
    def sampling_rate(self) -> float:

        return self.waveform.sampling_rate

    @property
    def frame_duration(self) -> float:
        
        # ToDo: Support frame duration
        return 1.

    @property
    def energy(self) -> float:

         # ToDo: Support frame energy
        return 1.0 

    def transmit(self, duration: float = 0.) -> Tuple[Signal]:

        if not self.waveform:
            raise RuntimeError("Radar waveform not specified")
        
        if not self.device:
            raise RuntimeError("Error attempting to transmit over a floating radar operator")

        # Generate the radar waveform
        signal = self.waveform.ping()
        
        # If the device has more than one antenna, a beamforming strategy is required
        if self.device.antennas.num_antennas > 1:
            
            # If no beamformer is configured, only the first antenna will transmit the ping
            if self.transmit_beamformer is None:

                additional_streams = Signal(np.zeros((self.device.antennas.num_antennas - signal.num_streams, signal.num_samples), dtype=complex), signal.sampling_rate)
                signal.append_streams(additional_streams)
                        
            elif self.transmit_beamformer.num_transmit_input_streams != 1:
                raise RuntimeError("Only transmit beamformers requiring a single input stream are supported by radar operators")
            
            elif self.transmit_beamformer.num_transmit_output_streams != self.device.antennas.num_antennas:
                raise RuntimeError("Radar operator transmit beamformers are required to consider the full number of antennas")
            
            else:
                signal = self.transmit_beamformer.transmit(signal)

        # Transmit signal over the occupied device slot (if the radar is attached to a device)
        if self._transmitter.attached:
            self._transmitter.slot.add_transmission(self._transmitter, signal)

        return signal,

    def receive(self) -> Tuple[RadarCube]:
        
        if not self.waveform:
            raise RuntimeError("Radar waveform not specified")
        
        if not self.device:
            raise RuntimeError("Error attempting to transmit over a floating radar operator")

        # Retrieve signal from receiver slot
        signal = self._receiver.signal.resample(self.waveform.sampling_rate)

        # If the device has more than one antenna, a beamforming strategy is required
        if self.device.antennas.num_antennas > 1:
            
            if self.receive_beamformer is None:
                raise RuntimeError("Receiving over a device with more than one antenna requires a beamforming configuration")
        
            if self.receive_beamformer.num_receive_output_streams != 1:
                raise RuntimeError("Only receive beamformers generating a single output stream are supported by radar operators")
            
            if self.receive_beamformer.num_receive_input_streams != self.device.antennas.num_antennas:
                raise RuntimeError("Radar operator receive beamformers are required to consider the full number of antenna streams")
            
            beamformed_samples = self.receive_beamformer.probe(signal)[:, 0, :]
            
        else:
            
            beamformed_samples = signal.samples

        # Build the radar cube by generating a beam-forming line over all angles of interest
        angles_of_interest = np.array([[0., 0.]], dtype=float) if self.receive_beamformer is None else self.receive_beamformer.probe_focus_points[:, 0, :]

        range_bins = self.waveform.range_bins
        velocity_bins = self.waveform.velocity_bins

        cube_data = np.empty((len(angles_of_interest),
                              len(velocity_bins),
                              len(range_bins)), dtype=float)

        for angle_idx, line in enumerate(beamformed_samples):

            # Process the single angular line by the waveform generator
            line_signal = Signal(line, signal.sampling_rate, carrier_frequency=signal.carrier_frequency)
            line_estimate = self.waveform.estimate(line_signal)

            cube_data[angle_idx, ::] = line_estimate

        # Create radar cube object
        cube = RadarCube(cube_data, angles_of_interest, velocity_bins, range_bins)

        return cube,
