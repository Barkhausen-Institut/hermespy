# -*- coding: utf-8 -*-
"""
====================
Audio Device Binding
====================

Hermes hardware bindings to audio devices offer the option to benchmark complex-valued
communication waveforms over affordable consumer-grade audio hardware.
The effective available bandwidth is limited to half of the audio devices sampling rate,
which is typically either :math:`44.1~\\mathrm{kHz}` or :math:`48~\\mathrm{kHz}`.


"""

from __future__ import annotations
from types import ModuleType
from typing import List, Optional, Union

import numpy as np
from scipy.fft import fft, ifft

from hermespy.core import Serializable, Signal, AntennaArrayBase
from ..physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AudioDeviceAntennas(AntennaArrayBase):
    """Antenna array information for audio devices."""

    __device: AudioDevice

    def __init__(self, device: AudioDevice) -> None:

        self.__device = device

    @property
    def num_antennas(self) -> int:

        return len(self.__device.playback_channels)

    @property
    def topology(self) -> np.ndarray:

        return np.zeros((len(self.__device.playback_channels), 3))

    def polarization(self, azimuth: float, elevation: float) -> np.ndarray:

        return np.sqrt(2) * np.ones((len(self.__device.playback_channels), 3))


class AudioDevice(PhysicalDevice, Serializable):
    """HermesPy binding to an arbitrary audio device. Let's rock!"""

    yaml_tag = "AudioDevice"
    property_blacklist = {"topology", "wavelength", "velocity", "orientation", "position", "random_mother"}

    __playback_device: int  # Device over which audio streams are to be transmitted
    __record_device: int  # Device over which audio streams are to be received
    __playback_channels: List[int]  # List of audio channel for signal transmission
    __record_channels: List[int]  # List of audio channel for signal reception
    __sampling_rate: float  # Configured sampling rate
    __transmission: Optional[np.ndarray]  # Configured transmission samples

    def __init__(self, playback_device: int, record_device: int, playback_channels: Union[List[int], None] = None, record_channels: Union[List[int], None] = None, sampling_rate: float = 48000, **kwargs) -> None:
        """
        Args:

            playback_device (int):
                Device over which audio streams are to be transmitted.

            record_device (int):
                Device over which audio streams are to be received.

            playback_channels (Union[np.ndarray, Iterable], optional):
                List of audio channels for signal transmission.
                By default, the first channel is selected.

            record_channels (Union[np.ndarray, Iterable], optional):
                List of audio channels for signal reception.
                By default, the first channel is selected.

            sampling_rate (float, optional):
                Configured sampling rate.
                48 kHz by default.
        """

        self.playback_device = playback_device
        self.record_device = record_device
        self.playback_channels = [1] if playback_channels is None else playback_channels
        self.record_channels = [1] if record_channels is None else record_channels
        self.sampling_rate = sampling_rate
        self.__transmission = None
        self.__reception = np.empty(0, float)

        antennas = AudioDeviceAntennas(self)
        PhysicalDevice.__init__(self, antennas=antennas, **kwargs)

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
    def playback_channels(self) -> List[int]:
        """Audio channels for signal transmission.

        Returns: List of audio channel indices.

        Raises:
            ValueError: On arguments not representing vectors.
        """

        return self.__playback_channels

    @playback_channels.setter
    def playback_channels(self, value: Union[np.ndarray, List[int]]):

        if isinstance(value, np.ndarray):
            self.__playback_channels = value.tolist()

        else:
            self.__playback_channels = value

    @property
    def record_channels(self) -> List[int]:
        """Audio channels for signal reception.

        Returns: List of audio channel indices.

        Raises:
            ValueError: On arguments not representing vectors.
        """

        return self.__record_channels

    @record_channels.setter
    def record_channels(self, value: Union[np.ndarray, List[int]]):

        if isinstance(value, np.ndarray):
            self.__record_channels = value.tolist()

        else:
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
        resampled_samples = signal.resample(self.sampling_rate).samples
        pressure_signal = np.roll(fft(resampled_samples), int(0.25 * resampled_samples.shape[1]), axis=1)
        pressure_signal[:, int(0.5 * pressure_signal.shape[1]) :] = 0.0
        pressure_signal = ifft(pressure_signal, axis=1).real
        pressure_signal = np.append(pressure_signal, np.zeros((pressure_signal.shape[0], delay_samples), dtype=float), axis=1)

        self.__transmission = pressure_signal.T
        self.__reception = np.empty((self.__transmission.shape[0], len(self.__record_channels)), dtype=float)

    def trigger(self) -> None:

        # Import sounddevice
        sd = self.__import_sd()

        # Simultaneously play and record samples
        sd.playrec(self.__transmission, self.sampling_rate, out=self.__reception, input_mapping=self.__record_channels, output_mapping=self.__playback_channels, device=(self.__record_device, self.__playback_device), blocking=False)

    def _download(self) -> Signal:

        # Import sounddevice
        sd = self.__import_sd()

        # Wait for recording and playing instructions to be finished
        sd.wait()

        # Scale reception to unit gain
        reception = self.__reception.T

        # Convert the received samples to complex using an implicit Hilbert transformation
        transform = fft(reception, axis=1)
        transform[:, int(0.5 * transform.shape[1]) :] = 0.0
        transform = np.roll(transform, -int(0.25 * transform.shape[1]), axis=1)
        complex_samples = ifft(2 * transform, axis=1)

        signal_model = Signal(complex_samples, self.sampling_rate, 0)
        return signal_model

    @staticmethod
    def __import_sd() -> ModuleType:
        """Import sounddevice.

        Required to prevent sounddevice from constantly messing with the system's audio settings.

        Returns: Sounddevice namespace module.
        """

        import sounddevice

        return sounddevice
