# -*- coding: utf-8 -*-
"""
================
Physical Devices
================
"""

from __future__ import annotations
from abc import abstractmethod
from pickle import dump, load
from time import sleep
from typing import List, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Device, Transmitter, Receiver
from hermespy.core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SilentTransmitter(Transmitter):
    """Silent transmitter mock."""

    __num_samples: int
    __sampling_rate: float

    def __init__(self,
                 num_samples: int,
                 sampling_rate: float) -> None:

        self.__num_samples = num_samples
        self.__sampling_rate = sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

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
        return self.__sampling_rate

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


class SignalTransmitter(Transmitter):
    """Silent transmitter mock."""

    __num_samples: int
    __sampling_rate: float

    def __init__(self,
                 num_samples: int,
                 sampling_rate: float) -> None:

        self.__num_samples = num_samples
        self.__sampling_rate = sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    def transmit(self,
                 signal: Signal) -> None:

        self.slot.add_transmission(self, signal)

    @property
    def frame_duration(self) -> float:
        return self.__num_samples / self.sampling_rate


class SignalReceiver(Receiver):
    """Noise receiver mock."""

    __num_samples: int
    __sampling_rate: float

    def __init__(self,
                 num_samples: int,
                 sampling_rate: float) -> None:

        self.__num_samples = num_samples
        self.__sampling_rate = sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def energy(self) -> float:
        return 0.

    def receive(self) -> Signal:
        return self.signal

    @property
    def frame_duration(self) -> float:
        return self.__num_samples / self.sampling_rate


class PhysicalDevice(Device):
    """Base representing any device controlling real hardware."""

    __calibration_delay: float
    __max_receive_delay: float

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

        self.__calibration_delay = 0.
        self.max_receive_delay = 0.

    @property
    def calibration_delay(self) -> float:
        """Delay compensation of the device calibration.

        Returns:
            float:
                Delay compensation in seconds.
        """

        return self.__calibration_delay

    @calibration_delay.setter
    def calibration_delay(self, vaue: float) -> None:

        self.__calibration_delay = vaue

    @abstractmethod
    def configure(self) -> None:
        ...

    @abstractmethod
    def trigger(self) -> None:
        """Trigger the device."""
        ...

    @abstractmethod
    def fetch(self) -> None:
        ...

    @property
    @abstractmethod
    def max_sampling_rate(self) -> float:
        ...

    @property
    def max_receive_delay(self) -> float:
        """Maximum expected delay during signal reception.

        The expected delay will be appended as additional samples to the expected frame length during reception.
        
        Returns:

            The expected receive delay in seconds.

        Raises:

            ValueError: If the delay is smaller than zero.
        """

        return self.__max_receive_delay

    @max_receive_delay.setter
    def max_receive_delay(self, value: float) ->  None:

        if value < 0.:
            raise ValueError("The maximum receive delay must be greater or equal to zero")

        self.__max_receive_delay = value

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

    def calibrate(self,
                  max_delay: float,
                  calibration_file: str,
                  num_iterations: int = 10,
                  wait: float = 0.) -> plt.Figure:
        """Calibrate the hardware.

        Currently the calibration will only compensate possible time-delays.
        Ideally, the transmit and receive channels of the device should be connected by a patch cable.
        WARNING: An attenuator element may be required! Be careful!!!!

        Args:

            max_delay (float):
                The maximum expected delay which the calibration should compensate for in seconds.

            calibration_file (str):
                Location of the calibration dump within the filesystem.

            num_iterations (int, optional):
                Number of calibration iterations.
                Default is 10.

            wait (float, optional):
                Idle time between iteration transmissions in seconds.

        Returns:

            plt.Figure:
                A figure of information regarding the calibration.
        """

        if num_iterations < 1:
            raise ValueError("The number of iterations must be greater or equal to one")

        if wait < 0.:
            raise ValueError("The waiting time must be greater or equal to zero")

        # Clear transmit caches
        self.transmitters.clear_cache()

        sampling_rate = self.max_sampling_rate
        num_samples = int(2 * max_delay * self.max_sampling_rate)
        if num_samples <= 1:
            raise ValueError("The assumed maximum delay is not resolvable by the configured sampling rate")

        # Register operators
        calibration_transmitter = SignalTransmitter(num_samples, sampling_rate)
        self.transmitters.add(calibration_transmitter)

        calibration_receiver = SignalReceiver(num_samples, sampling_rate)
        self.receivers.add(calibration_receiver)

        dirac_index = int(max_delay * sampling_rate)
        waveform = np.zeros((self.num_antennas, num_samples), dtype=complex)
        waveform[:, dirac_index] = 1.
        calibration_signal = Signal(waveform, sampling_rate, self.carrier_frequency)

        propagated_signals: List[Signal] = []
        propagated_dirac_indices = np.empty(num_iterations, dtype=int)

        # Make multiple iteration calls for calibration
        for i in range(num_iterations):

            # Send and receive calibration waveform
            calibration_transmitter.transmit(calibration_signal)

            self.configure()
            self.trigger()
            self.fetch()

            propagated_signal = calibration_receiver.receive()

            # Infer the implicit delay by estimating the sample index of the propagated dirac
            propagated_signals.append(propagated_signal)
            propagated_dirac_indices = np.argmax(abs(propagated_signal.samples[0, :]))

            # Wait the configured amount of time between iterations
            sleep(wait)

        # Un-register operators
        self.transmitters.remove(calibration_transmitter)
        self.receivers.remove(calibration_receiver)

        # Compute calibration delay
        mean_dirac_index = np.mean(propagated_dirac_indices)   # This is just a ML estimation
        calibration_delay = (dirac_index - mean_dirac_index) / propagated_signal.sampling_rate

        # Dump calibration to pickle savefile
        with open(calibration_file, 'wb') as file_stream:
            dump(calibration_delay, file_stream)

        # Save register calibration delay internally
        self.__calibration_delay = calibration_delay

        # Visualize the results
        fig, axes = plt.subplots(2, 1)
        fig.suptitle('Device Delay Calibration')

        axes[0].plot(calibration_signal.timestamps, abs(calibration_signal.samples[0, :]))
        for sig in propagated_signals:
            axes[1].plot(sig.timestamps, abs(sig.samples[0, :]), color='blue')
        axes[1].axvline(x=(dirac_index / sampling_rate - calibration_delay), color='red')

    def load_calibration(self,
                         calibration_file: str) -> None:
        """Recall a dumped calibration from the filesystem.

        Args:
            calibration_file (str):
                Location of the calibration dump.
        """

        with open(calibration_file, 'rb') as file_stream:
            self.__calibration_delay = load(file_stream)


PhysicalDeviceType = TypeVar('PhysicalDevicType', bound=PhysicalDevice)
"""Type of phyiscal device."""
