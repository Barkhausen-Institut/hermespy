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
from typing import Optional, Type

import numpy as np
from h5py import Group

from hermespy.beamforming import ReceiveBeamformer, TransmitBeamformer
from hermespy.core import ChannelStateInformation, DuplexOperator, Signal, Serializable, SNRType, Transmission, Reception, ReceptionType
from .cube import RadarCube
from .detection import RadarDetector, RadarPointCloud

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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
    def estimate(self, signal: Signal) -> np.ndarray:
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
            np.ndarray: Doppler shift in Hz.
        """
        ...  # pragma no cover

    @property
    @abstractmethod
    def energy(self) -> float:
        """Energy of the radar waveform.

        Returns: Radar energy in :math:`\\mathrm{Wh}`.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> float:
        """Power of the radar waveform.

        Returns: Radar power in :math:`\\mathrm{W}`.
        """
        ...  # pragma: no cover


class RadarTransmission(Transmission):
    """Information generated by transmitting over a radar operator."""

    def __init__(self, signal: Signal) -> None:
        """
        Args:

            signal (Signal): Transmitted radar waveform.
        """

        Transmission.__init__(self, signal)


class RadarReception(Reception, RadarCube):
    """Information generated by receiving over a radar operator."""

    cube: RadarCube
    """Processed raw radar data."""

    cloud: Optional[RadarPointCloud]
    """Radar point cloud."""

    def __init__(self, signal: Signal, cube: RadarCube, cloud: Optional[RadarPointCloud] = None) -> None:
        """
        Args:

            signal (Signal): Received radar waveform.
            cube (RadarCube): Processed raw radar data.
            cloud (RadarPointCloud, optional): Radar point cloud.
        """

        Reception.__init__(self, signal)

        self.cube = cube
        self.cloud = cloud

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        Reception.to_HDF(self, group)

        # Serialize class attributes
        self.cube.to_HDF(self._create_group(group, "cube"))
        return

    @classmethod
    def from_HDF(cls: Type[RadarReception], group: Group) -> RadarReception:
        signal = Signal.from_HDF(group["signal"])
        cube = RadarCube.from_HDF(group["cube"])

        return RadarReception(signal, cube)


class Radar(DuplexOperator[RadarTransmission, RadarReception], Serializable):
    """HermesPy representation of a mono-static radar sensing its environment."""

    yaml_tag = "Radar"
    property_blacklist = {"slot"}

    __transmit_beamformer: Optional[TransmitBeamformer]
    __receive_beamformer: Optional[ReceiveBeamformer]
    __waveform: Optional[RadarWaveform]
    __detector: Optional[RadarDetector]

    def __init__(self) -> None:
        self.waveform = None
        self.receive_beamformer = None
        self.transmit_beamformer = None
        self.__waveform = None
        self.detector = None

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
        return 1.0

    def _noise_power(self, strength: float, snr_type=SNRType) -> float:
        # No waveform configured equals no noise required
        if self.waveform is None:
            return 0.0

        if snr_type == SNRType.EN0:
            return self.waveform.energy / strength

        if snr_type == SNRType.PN0:
            return self.waveform.power / strength

        raise ValueError(f"SNR of type '{snr_type}' is not supported by radar operators")

    @property
    def waveform(self) -> Optional[RadarWaveform]:
        """The waveform to be emitted by this radar.

        Returns:

            The configured waveform.
            `None` if the waveform is undefined.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: Optional[RadarWaveform]) -> None:
        self.__waveform = value

    @property
    def detector(self) -> Optional[RadarDetector]:
        """The detector configured to process the resulting radar cube.

        Returns:

            The configured detector.
            `None` if the detector is undefined.
        """

        return self.__detector

    @detector.setter
    def detector(self, value: Optional[RadarDetector]) -> None:
        self.__detector = value

    def _transmit(self, duration: float = 0.0) -> RadarTransmission:
        if not self.__waveform:
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

        # Prepare transmission
        signal.carrier_frequency = self.carrier_frequency
        transmission = RadarTransmission(signal)

        return transmission

    def _receive(self, signal: Signal, _: ChannelStateInformation) -> RadarReception:
        if not self.waveform:
            raise RuntimeError("Radar waveform not specified")

        if not self.device:
            raise RuntimeError("Error attempting to receive over a floating radar operator")

        # Resample signal properly
        signal = signal.resample(self.__waveform.sampling_rate)

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
        angles_of_interest = np.array([[0.0, 0.0]], dtype=float) if self.receive_beamformer is None else self.receive_beamformer.probe_focus_points[:, 0, :]

        range_bins = self.waveform.range_bins
        velocity_bins = self.waveform.velocity_bins

        cube_data = np.empty((len(angles_of_interest), len(velocity_bins), len(range_bins)), dtype=float)

        for angle_idx, line in enumerate(beamformed_samples):
            # Process the single angular line by the waveform generator
            line_signal = Signal(line, signal.sampling_rate, carrier_frequency=signal.carrier_frequency)
            line_estimate = self.waveform.estimate(line_signal)

            cube_data[angle_idx, ::] = line_estimate

        # Create radar cube object
        cube = RadarCube(cube_data, angles_of_interest, velocity_bins, range_bins)

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        reception = RadarReception(signal, cube, cloud)
        return reception

    def _recall_transmission(self, group: Group) -> RadarTransmission:
        return RadarTransmission.from_HDF(group)

    def _recall_reception(self, group: Group) -> RadarReception:
        return RadarReception.from_HDF(group)
