# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from types import ModuleType
from typing import Sequence
from typing_extensions import override

import numpy as np
from scipy.fft import fft, ifft

from hermespy.core import (
    AntennaMode,
    Serializable,
    Signal,
    Antenna,
    AntennaArray,
    AntennaPort,
    SerializationProcess,
    DeserializationProcess,
)
from ..physical_device import PhysicalDevice, PhysicalDeviceState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AudioAntenna(Antenna):
    """Antenna model for audio devices."""

    def copy(self) -> AudioAntenna:
        return AudioAntenna(self.mode, self.pose)

    def local_characteristics(self, azimuth: float, elevation) -> np.ndarray:
        return np.array([2**0.5, 2**0.5], dtype=float)


class AudioPort(ABC, AntennaPort[AudioAntenna, "AudioDeviceAntennas"]):
    """Antenna port model for audio devices."""

    @property
    @abstractmethod
    def channel(self) -> int:
        """Audio channel index."""
        ...  # pragma: no cover


class AudioPlaybackPort(AudioPort):
    """Antenna transmit port model for audio devices.

    Represents a single transmitting audio channel of a soundcard.
    """

    __channel: int

    def __init__(self, array: AudioDeviceAntennas, channel: int) -> None:
        """
        Args:

            array:
                Antenna array to which the port belongs.

            channel:
                Recording audio channel index.
        """

        # Save attributes
        self.__channel = channel

        # Initialize base class
        antennas = [AudioAntenna(AntennaMode.TX)]
        AntennaPort.__init__(self, antennas, None, array)

    @property
    def channel(self) -> int:
        """Recording audio channel index."""

        return self.__channel


class AudioRecordPort(AudioPort):
    """Antenna receive port model for audio devices.

    Represents a single receiving audio channel of a soundcard.
    """

    def __init__(self, array: AudioDeviceAntennas, channel: int) -> None:
        """
        Args:

            array:
                Antenna array to which the port belongs.

            channel:
                Recording audio channel index.
        """

        # Initialize base class
        antennas = [AudioAntenna(AntennaMode.RX)]
        AntennaPort.__init__(self, antennas, None, array)

        # Save attributes
        self.__channel = channel

    @property
    def channel(self) -> int:
        """Recording audio channel index."""

        return self.__channel


class AudioDeviceAntennas(AntennaArray[AudioPort, AudioAntenna]):
    """Antenna array information for audio devices."""

    __device: AudioDevice

    def __init__(self, device: AudioDevice) -> None:
        # Initialize base class
        AntennaArray.__init__(self)

        # Save attributes
        self.__device = device

        # Create ports
        self.__transmit_ports = [
            AudioPlaybackPort(self, channel) for channel in self.__device.playback_channels
        ]
        self.__receiver_ports = [
            AudioRecordPort(self, channel) for channel in self.__device.record_channels
        ]

        # Configure the kinematics
        self.set_base(device)

    def _new_port(self) -> AudioPort:
        raise RuntimeError("Audio antenna arrays can't create new ports")

    @property
    def ports(self) -> Sequence[AudioPort]:
        return [*self.__transmit_ports, *self.__receiver_ports]

    @property
    def transmit_ports(self) -> Sequence[AudioPlaybackPort]:
        return self.__transmit_ports

    @property
    def receive_ports(self) -> Sequence[AudioRecordPort]:
        return self.__receiver_ports

    @property
    def num_antennas(self) -> int:
        return len(self.__transmit_ports) + len(self.__receiver_ports)

    @property
    def num_receive_antennas(self) -> int:
        return len(self.__device.playback_channels)

    @property
    def num_transmit_antennas(self) -> int:
        return len(self.__device.record_channels)


class AudioDevice(PhysicalDevice[PhysicalDeviceState], Serializable):
    """HermesPy binding to an arbitrary audio device. Let's rock!"""

    __DEFAULT_SAMPLING_RATE = 48000.0

    __playback_device: int  # Device over which audio streams are to be transmitted
    __record_device: int  # Device over which audio streams are to be received
    __playback_channels: Sequence[int]  # List of audio channel for signal transmission
    __record_channels: Sequence[int]  # List of audio channel for signal reception
    __sampling_rate: float  # Configured sampling rate
    __transmission: np.ndarray | None  # Configured transmission samples
    __antennas: AudioDeviceAntennas  # Antenna array information

    def __init__(
        self,
        playback_device: int,
        record_device: int,
        playback_channels: Sequence[int] | None = None,
        record_channels: Sequence[int] | None = None,
        sampling_rate: float = __DEFAULT_SAMPLING_RATE,
        **kwargs,
    ) -> None:
        """
        Args:

            playback_device:
                Device over which audio streams are to be transmitted.

            record_device:
                Device over which audio streams are to be received.

            playback_channels:
                List of audio channels for signal transmission.
                By default, the first channel is selected.

            record_channels:
                List of audio channels for signal reception.
                By default, the first channel is selected.

            sampling_rate:
                Configured sampling rate.
                48 kHz by default.
        """

        # Initialize base class
        PhysicalDevice.__init__(self, **kwargs)

        # Initialize attributes
        self.playback_device = playback_device
        self.record_device = record_device
        self.playback_channels = [1] if playback_channels is None else playback_channels
        self.record_channels = [1] if record_channels is None else record_channels
        self.sampling_rate = sampling_rate
        self.__transmission = None
        self.__reception = np.empty(0, float)
        self.__antennas = AudioDeviceAntennas(self)

    def state(self) -> PhysicalDeviceState:
        return PhysicalDeviceState(
            id(self),
            datetime.now().timestamp(),
            self.pose,
            self.velocity,
            self.carrier_frequency,
            self.sampling_rate,
            self.num_digital_transmit_ports,
            self.num_digital_receive_ports,
            self.antennas.state(self.pose),
        )

    @property
    def antennas(self) -> AudioDeviceAntennas:
        return self.__antennas

    @property
    def playback_device(self) -> int:
        """Device over which audio streams are to be transmitted.

        Returns: Device identifier.

        Raises:
            ValueError: For negative identifiers.
        """

        return self.__playback_device

    @playback_device.setter
    def playback_device(self, value: int) -> None:
        if value < 0:
            raise ValueError("Playback device identifier must be greater or equal to zero")

        self.__playback_device = value

    @property
    def record_device(self) -> int:
        """Device over which audio streams are to be received.

        Returns: Device identifier.

        Raises:
            ValueError: For negative identifiers.
        """

        return self.__record_device

    @record_device.setter
    def record_device(self, value: int) -> None:
        if value < 0:
            raise ValueError("Record device identifier must be greater or equal to zero")

        self.__record_device = value

    @property
    def playback_channels(self) -> Sequence[int]:
        """Audio channels for signal transmission.

        Returns: Sequence of audio channel indices.

        Raises:
            ValueError: On arguments not representing vectors.
        """

        return self.__playback_channels

    @playback_channels.setter
    def playback_channels(self, value: Sequence[int]):
        self.__playback_channels = value

    @property
    def record_channels(self) -> Sequence[int]:
        """Audio channels for signal reception.

        Returns: Sequence of audio channel indices.

        Raises:
            ValueError: On arguments not representing vectors.
        """

        return self.__record_channels

    @record_channels.setter
    def record_channels(self, value: Sequence[int]):
        self.__record_channels = value

    @property
    def carrier_frequency(self) -> float:
        return 0.5 * self.sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate

    def _upload(self, signal: Signal) -> None:
        # Infer parameters
        delay_samples = int(self.max_receive_delay * self.sampling_rate)

        # Mix to to positive frequencies for audio transmission
        resampled_samples = signal.resample(self.sampling_rate).getitem()
        pressure_signal = np.roll(
            fft(resampled_samples), int(0.25 * resampled_samples.shape[1]), axis=1
        )
        pressure_signal[:, int(0.5 * pressure_signal.shape[1]) :] = 0.0
        pressure_signal = ifft(pressure_signal, axis=1).real
        pressure_signal = np.append(
            pressure_signal,
            np.zeros((pressure_signal.shape[0], delay_samples), dtype=float),
            axis=1,
        )

        self.__transmission = pressure_signal.T
        self.__reception = np.empty(
            (self.__transmission.shape[0], len(self.__record_channels)), dtype=float
        )

    def trigger(self) -> None:
        # Import sounddevice
        sd = self._import_sd()

        # Simultaneously play and record samples
        sd.playrec(
            self.__transmission,
            self.sampling_rate,
            out=self.__reception,
            input_mapping=self.__record_channels,
            output_mapping=self.__playback_channels,
            device=(self.__record_device, self.__playback_device),
            blocking=False,
        )

    def _download(self) -> Signal:
        # Import sounddevice
        sd = self._import_sd()

        # Wait for recording and playing instructions to be finished
        sd.wait()

        # Scale reception to unit gain
        reception = self.__reception.T

        # Convert the received samples to complex using an implicit Hilbert transformation
        transform = fft(reception, axis=1)
        transform[:, int(0.5 * transform.shape[1]) :] = 0.0
        transform = np.roll(transform, -int(0.25 * transform.shape[1]), axis=1)
        complex128samples = ifft(2 * transform, axis=1)

        signal_model = Signal.Create(complex128samples, self.sampling_rate, 0)
        return signal_model

    @staticmethod
    def _import_sd() -> ModuleType:
        """Import sounddevice.

        Required to prevent sounddevice from constantly messing with the system's audio settings.

        Returns: Sounddevice namespace module.
        """

        import sounddevice

        return sounddevice

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.playback_device, "playback_device")
        process.serialize_integer(self.record_device, "record_device")
        process.serialize_array(np.asarray(self.playback_channels), "playback_channels")
        process.serialize_array(np.asarray(self.record_channels), "record_channels")
        process.serialize_floating(self.sampling_rate, "sampling_rate")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> AudioDevice:
        return cls(
            process.deserialize_integer("playback_device"),
            process.deserialize_integer("record_device"),
            process.deserialize_array("playback_channels", np.int64).tolist(),
            process.deserialize_array("record_channels", np.int64).tolist(),
            process.deserialize_floating("sampling_rate", cls.__DEFAULT_SAMPLING_RATE),
        )
