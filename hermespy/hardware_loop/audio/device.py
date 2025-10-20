# -*- coding: utf-8 -*-

from __future__ import annotations
from datetime import datetime
from itertools import chain
from types import ModuleType
from typing import Sequence
from typing_extensions import override

import numpy as np
from scipy.fft import fft, ifft

from hermespy.core import (
    AntennaMode,
    DenseSignal,
    Serializable,
    Signal,
    Antenna,
    AntennaArray,
    SerializationProcess,
    DeserializationProcess,
)
from ..physical_device import PhysicalDevice, PhysicalDeviceState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AudioAntenna(Antenna):
    """Antenna model for audio devices."""

    def copy(self) -> AudioAntenna:
        return AudioAntenna(self.mode, self.pose)

    def local_characteristics(self, azimuth: float, elevation) -> np.ndarray:
        return np.array([2**0.5, 2**0.5], dtype=float)


class AudioDeviceAntennas(AntennaArray[AudioAntenna]):
    """Antenna array information for audio devices."""

    __device: AudioDevice
    __transmit_antennas: list[AudioAntenna]
    __receive_antennas: list[AudioAntenna]

    def __init__(self, device: AudioDevice) -> None:
        """
        Args:
            device: The audio device to which the antenna array belongs.
        """

        # Initialize base class
        AntennaArray.__init__(self)

        # Save attributes
        self.__device = device

        # Create ports
        self.__transmit_antennas = [
            AudioAntenna(AntennaMode.TX) for _ in self.__device.playback_channels
        ]
        self.__receive_antennas = [
            AudioAntenna(AntennaMode.RX) for _ in self.__device.record_channels
        ]

        for antenna in chain(self.__transmit_antennas, self.__receive_antennas):
            antenna.set_base(self)

        # Configure the kinematics
        self.set_base(device)

    @property
    @override
    def num_receive_antennas(self) -> int:
        return len(self.__device.playback_channels)

    @property
    @override
    def num_transmit_antennas(self) -> int:
        return len(self.__device.record_channels)

    @property
    @override
    def transmit_antennas(self) -> list[AudioAntenna]:
        return self.__transmit_antennas

    @property
    @override
    def receive_antennas(self) -> list[AudioAntenna]:
        return self.__receive_antennas

    @property
    @override
    def antennas(self) -> list[AudioAntenna]:
        return self.__transmit_antennas + self.__receive_antennas


class AudioDevice(PhysicalDevice[PhysicalDeviceState], Serializable):
    """HermesPy binding to an arbitrary audio device. Let's rock!"""

    __DEFAULT_SAMPLING_RATE = 48000.0
    __DEFAULT_OVERSAMPLING_FACTOR = 2

    __playback_device: int  # Device over which audio streams are to be transmitted
    __record_device: int  # Device over which audio streams are to be received
    __playback_channels: Sequence[int]  # List of audio channel for signal transmission
    __record_channels: Sequence[int]  # List of audio channel for signal reception
    __sampling_rate: float  # Configured sampling rate
    __oversampling_factor: int  # Oversampling factor for the internal signal processing
    __reception: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    __transmission: np.ndarray | None  # Configured transmission samples
    __antennas: AudioDeviceAntennas  # Antenna array information

    def __init__(
        self,
        playback_device: int,
        record_device: int,
        playback_channels: Sequence[int] | None = None,
        record_channels: Sequence[int] | None = None,
        sampling_rate: float = __DEFAULT_SAMPLING_RATE,
        oversampling_factor: int = __DEFAULT_OVERSAMPLING_FACTOR,
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

            oversampling_factor:
                Oversampling factor for the internal signal processing.
                1 by default.
        """

        # Initialize base class
        PhysicalDevice.__init__(self, **kwargs)

        # Initialize attributes
        self.playback_device = playback_device
        self.record_device = record_device
        self.playback_channels = [1] if playback_channels is None else playback_channels
        self.record_channels = [1] if record_channels is None else record_channels
        self.sampling_rate = sampling_rate
        self.oversampling_factor = oversampling_factor
        self.__transmission = None
        self.__reception = np.empty((0, 0), dtype=np.float64)
        self.__antennas = AudioDeviceAntennas(self)

    def state(self) -> PhysicalDeviceState:
        return PhysicalDeviceState(
            id(self),
            datetime.now().timestamp(),
            self.pose,
            self.velocity,
            self.carrier_frequency,
            self.sampling_rate / self.oversampling_factor,
            self.oversampling_factor,
            self.antennas.state(self.pose),
            self.num_transmit_dsp_ports,
            self.num_receive_dsp_ports,
            self.num_transmit_rf_ports,
            self.num_receive_rf_ports,
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
    def oversampling_factor(self) -> int:
        """Oversampling factor for the internal signal processing.

        Raises:
            ValueError: For values smaller than one.
        """
        return self.__oversampling_factor

    @oversampling_factor.setter
    def oversampling_factor(self, value: int) -> None:
        if value < 1:
            raise ValueError("Oversampling factor must be a positive integer")
        self.__oversampling_factor = value

    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate

    @override
    def _upload(self, signal: Signal) -> DenseSignal:
        # Infer parameters
        delay_samples = int(self.max_receive_delay * self.sampling_rate)

        # Mix to to positive frequencies for audio transmission
        pressure_spectrum = np.roll(
            np.asarray(fft(signal.to_dense()), np.complex128),
            int(0.25 * signal.num_samples) + 1,
            axis=1,
        ).reshape((self.num_transmit_rf_ports, -1))
        pressure_spectrum[:, int(0.5 * pressure_spectrum.shape[1]) :] = 0.0
        real_pressure_signal = np.asarray(ifft(pressure_spectrum, axis=1), np.complex128).real.reshape(
            (self.num_transmit_rf_ports, -1)
        )
        real_pressure_signal = np.append(
            real_pressure_signal,
            np.zeros((real_pressure_signal.shape[0], delay_samples), dtype=np.float64),
            axis=1,
        ).reshape((self.num_transmit_rf_ports, -1))

        self.__transmission = real_pressure_signal.T
        self.__reception = np.empty(
            (self.__transmission.shape[0], len(self.__record_channels)), dtype=np.float64
        )
        return DenseSignal.FromNDArray(
            real_pressure_signal.reshape((self.num_receive_rf_ports, -1)).astype(np.complex128),
            signal.sampling_rate,
            0.0,
        )

    @override
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

    @override
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
        transform = np.roll(transform, -int(0.25 * transform.shape[1])-1, axis=1)
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
        process.serialize_integer(self.oversampling_factor, "oversampling_factor")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> AudioDevice:
        return cls(
            process.deserialize_integer("playback_device"),
            process.deserialize_integer("record_device"),
            process.deserialize_array("playback_channels", np.int64).tolist(),
            process.deserialize_array("record_channels", np.int64).tolist(),
            process.deserialize_floating("sampling_rate", cls.__DEFAULT_SAMPLING_RATE),
            process.deserialize_integer("oversampling_factor", cls.__DEFAULT_OVERSAMPLING_FACTOR),
        )
