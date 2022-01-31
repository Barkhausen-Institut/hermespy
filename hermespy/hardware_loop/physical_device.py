# -*- coding: utf-8 -*-
"""
================
Physical Devices
================
"""

from __future__ import annotations

import numpy as np
from abc import abstractmethod
from typing import Tuple

from hermespy.core import Device, Transmitter, Receiver
from hermespy.core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SilentTransmitter(Transmitter):
    """Silent transmitter mock."""

    __num_samples: int

    def __init__(self,
                 num_samples: int) -> None:

        self.__num_samples = num_samples

    @property
    def sampling_rate(self) -> float:
        return self.device.sampling_rate

    def transmit(self,
                 duration: float = 0.) -> Tuple[Signal]:

        silence = Signal(np.zeros((self.device.num_antennas, self.__num_samples), dtype=complex),
                         sampling_rate=self.sampling_rate,
                         carrier_frequency=self.device.carrier_frequency)

        self.slot.add_transmission(self, silence)
        return silence,

    @property
    def frame_duration(self) -> float:
        return self.__num_samples / self.sampling_rate


class PowerReceiver(Receiver):
    """Noise receiver mock."""

    @property
    def sampling_rate(self) -> float:
        return self.device.sampling_rate

    @property
    def energy(self) -> float:
        return 0.

    def receive(self) -> Tuple[np.ndarray]:

        # Fetch noise samples
        signal = self.signal.samples

        # Compute signal power
        power = np.var(signal, axis=1)

        return power,

    @property
    def frame_duration(self) -> float:
        return 0.


class PhysicalDevice(Device):
    """Base representing any device controlling real hardware."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

    @abstractmethod
    def trigger(self) -> None:
        """Trigger the device."""
        ...

    def receive(self, signal: Signal) -> None:
        """Receive a new signal at this physical device.

        Args:
            signal (Signal):
                Signal model to be received.
        """

        # Signal is now a baseband-signal
        signal.carrier_frequency = 0.

        for receiver in self.receivers:

            receiver.cache_reception(signal)

    @property
    def velocity(self) -> np.ndarray:

        raise NotImplementedError("The velocity of physical devices is undefined by default")

    def estimate_noise_power(self,
                             num_samples: int = 1000) -> np.ndarray:
        """Estimate the power of the noise floor within the hardware.

        Receives `num_samples` without any transmissions and estimates the power over them.
        Note that any interference in this case will be interpreted as noise.

        Args:

            num_samples (int, optional):
                Number of received samples.
                1000 by default.

        Returns:
            np.ndarray:
                Estimated noise power at each respective receive channel,
                i.e. the variance of the unbiased received samples.
        """

        # Clear transmit caches
        self.transmitters.clear_cache()

        # Register a new virtual transmitter only for the purpose of transmitting nothing
        silent_transmitter = SilentTransmitter(num_samples)
        self.transmitters.add(silent_transmitter)

        # Register a new virtual receiver for the purpose of receiving the hardware noise
        power_receiver = PowerReceiver()
        self.receivers.add(power_receiver)

        # Receive noise floor
        _ = silent_transmitter.transmit()
        self.trigger()
        noise_power, = power_receiver.receive()

        return noise_power
