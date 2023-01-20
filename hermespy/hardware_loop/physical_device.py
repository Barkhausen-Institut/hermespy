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
from typing import List, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt

from hermespy.core import Device, Transmitter, Receiver
from hermespy.core.device import Reception, Transmission
from hermespy.core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class StaticOperator(object):
    """Base class for static device operators"""

    __num_samples: int  # Number of samples per transmission
    __sampling_rate: float  # Sampling rate of transmission

    def __init__(self, num_samples: int, sampling_rate: float) -> None:
        """
        Args:

            num_samples (int):
                Number of samples per transmission.

            sampling_rate (float):
                Sampling rate of transmission.
        """

        self.__num_samples = num_samples
        self.__sampling_rate = sampling_rate

    @property
    def num_samples(self) -> int:
        """Number of samples per transmission.

        Returns: Number of samples.
        """

        return self.__num_samples

    @property
    def sampling_rate(self) -> float:

        return self.__sampling_rate

    @property
    def frame_duration(self) -> float:

        return self.__num_samples / self.sampling_rate


class SilentTransmitter(StaticOperator, Transmitter):
    """Silent transmitter mock."""

    def transmit(self, duration: float = 0.0) -> Transmission:

        # Compute the number of samples to be transmitted
        num_samples = self.num_samples if duration <= 0.0 else int(duration * self.sampling_rate)

        silence = Signal(np.zeros((self.device.num_antennas, num_samples), dtype=complex), sampling_rate=self.sampling_rate, carrier_frequency=self.device.carrier_frequency)

        transmission = Transmission(silence)

        self.device.transmitters.add_transmission(self, transmission)
        return transmission


class SignalTransmitter(StaticOperator, Transmitter):
    """Custom signal transmitter."""

    def transmit(self, signal: Signal) -> Transmission:

        transmission = Transmission(signal)

        self.device.transmitters.add_transmission(self, transmission)
        return transmission


class PowerReceiver(Receiver):
    """Noise power receiver for the device noise floor estimation routine."""

    __num_samples: int

    def __init__(self, num_samples: int) -> None:
        """
        Args:
        
            num_samples (int):
                Number of samples required for power estimation.
        """
        
        if num_samples <= 1:
            raise ValueError("Number of required samples must be greater or equal to one")
        
        self.__num_samples = num_samples
        
    @property
    def num_samples(self) -> int:
        """Number of samples required for power estimation.
        
        Returns: Number of samples.
        """
        
        return self.__num_samples

    @property
    def sampling_rate(self) -> float:

        return self.device.sampling_rate

    @property
    def energy(self) -> float:

        return 0.0

    def receive(self) -> Reception:

        # Fetch noise samples
        signal = self.signal
        return Reception(signal)

    @property
    def frame_duration(self) -> float:

        return self.num_samples * self.sampling_rate


class SignalReceiver(StaticOperator, Receiver):
    """Custom signal receiver."""

    @property
    def energy(self) -> float:

        return 0.0

    def receive(self) -> Reception:

        received_signal = self.signal.resample(self.sampling_rate)
        return Reception(received_signal)


class PhysicalDevice(Device):
    """Base representing any device controlling real hardware."""

    __calibration_delay: float
    __max_receive_delay: float
    __adaptive_sampling: bool
    __lowpass_filter: bool
    __lowpass_bandwidth: float

    def __init__(self, max_receive_delay: float = 0.0, calibration_delay: float = 0.0, *args, **kwargs) -> None:
        """
        Args:
            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

        self.calibration_delay = calibration_delay
        self.max_receive_delay = max_receive_delay
        self.__adaptive_sampling = False
        self.__lowpass_filter = False
        self.__lowpass_bandwidth = 0.0

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
    def trigger(self) -> None:
        """Trigger the device."""
        ...  # pragma no cover

    @property
    def adaptive_sampling(self) -> bool:
        """Allow adaptive sampling during transmission.

        Returns: Enabled flag.
        """

        return self.__adaptive_sampling

    @adaptive_sampling.setter
    def adaptive_sampling(self, value: bool) -> None:

        self.__adaptive_sampling = bool(value)

    @property
    def lowpass_filter(self) -> bool:
        """Apply a digital lowpass filter to the received base-band samples.

        Returns: Enabled flag.
        """

        return self.__lowpass_filter

    @lowpass_filter.setter
    def lowpass_filter(self, value: bool) -> None:

        self.__lowpass_filter = bool(value)

    @property
    def lowpass_bandwidth(self) -> float:
        """Digital lowpass filter bandwidth

        Returns:
            Filter bandwidth in Hz.
            Zero if the device should determine the bandwidth automatically.

        Raises:

            ValueError: For negative bandwidths.
        """

        return self.__lowpass_bandwidth

    @lowpass_bandwidth.setter
    def lowpass_bandwidth(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("Lowpass filter bandwidth should be greater or equal to zero")

        self.__lowpass_bandwidth = float(value)

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
    def max_receive_delay(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("The maximum receive delay must be greater or equal to zero")

        self.__max_receive_delay = value

    @property
    @abstractmethod
    def max_sampling_rate(self) -> float:
        """Maximal device sampling rate.

        Returns: The samplin rate in Hz.
        """

        ...  # pragma no cover

    @property
    def velocity(self) -> np.ndarray:

        raise NotImplementedError("The velocity of physical devices is undefined by default")

    def estimate_noise_power(self, num_samples: int = 1000) -> np.ndarray:
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
        silent_transmitter = SilentTransmitter(num_samples, self.sampling_rate)
        self.transmitters.add(silent_transmitter)

        # Register a new virtual receiver for the purpose of receiving the hardware noise
        power_receiver = PowerReceiver(num_samples)
        self.receivers.add(power_receiver)

        # Receive noise floor
        _ = silent_transmitter.transmit()

        self.transmit()
        self.trigger()
        self.receive()

        reception = power_receiver.receive()

        return reception.signal.power

    def _upload(self, signal: Signal) -> None:
        """Upload samples to be transmitted to the represented hardware.

        Args:

            signal (Signal):
                The samples to be uploaded.
        """

        # The default routine is a stub
        return  # pragma no cover

    def transmit(self, clear_cache: bool = True) -> Signal:

        # If adaptive sampling is disabled, resort to the default transmission routine
        if not self.adaptive_sampling or self.transmitters.num_operators < 1:

            device_transmission = Device.transmit(self, clear_cache)

        else:

            transmissions = self.transmitters.get_transmissions(clear_cache)

            device_transmission = transmissions[0].signal
            if device_transmission is None:
                return None

            for transmission in transmissions[1:]:

                if transmission is None:
                    continue

                if transmission.signal.sampling_rate != device_transmission.sampling_rate:
                    raise RuntimeError("Adpative sampling does not support operators with differing sampling rates")

                device_transmission.superimpose(transmission.signal, resample=False)

            # Adaptive sampling rate will simply assume the device's sampling rate
            device_transmission.sampling_rate = self.sampling_rate
            device_transmission.carrier_frequency = self.carrier_frequency

        # Upload the samples
        self._upload(device_transmission)

        # Return samples
        return device_transmission

    def _download(self) -> Signal:
        """Download received samples from the represented hardware.

        Returns:
            A signal model of the downloaded samples.
        """

        # This method is a stub by default
        raise NotImplementedError("The default physical device does not support downloads")

    def receive(self, signal: Optional[Signal] = None) -> Signal:

        if signal is None:
            signal = self._download()

        if signal.num_streams != self.num_antennas:
            raise ValueError("Number of received signal streams does not match number of configured antennas")

        if signal.sampling_rate != self.sampling_rate:
            raise ValueError(f"Received signal sampling rate ({signal.sampling_rate}) does not match device sampling rate({self.sampling_rate})")

        filtered_signal = signal.copy()

        if self.lowpass_filter:

            if self.lowpass_bandwidth <= 0.0:
                filter_cutoff = 0.25 * self.sampling_rate

            else:
                filter_cutoff = 0.5 * self.lowpass_bandwidth

            filter = butter(5, filter_cutoff, output="sos", fs=self.sampling_rate)
            filtered_signal.samples = sosfilt(filter, filtered_signal.samples, axis=1)

        # Cache the reception at all operators
        for receiver in self.receivers:

            received_signal = filtered_signal.copy()
            received_signal.carrier_frequency = 0.0

            # Simply override the sampling rate if adaptive sampling is enabled
            if self.adaptive_sampling:
                received_signal.sampling_rate = receiver.sampling_rate

            receiver.cache_reception(received_signal)

        return filtered_signal

    def calibrate(self, max_delay: float, calibration_file: str, num_iterations: int = 10, wait: float = 0.0) -> plt.Figure:
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

        if wait < 0.0:
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
        waveform[:, dirac_index] = 1.0
        calibration_signal = Signal(waveform, sampling_rate, self.carrier_frequency)

        propagated_signals: List[Signal] = []
        propagated_dirac_indices = np.empty(num_iterations, dtype=int)

        # Make multiple iteration calls for calibration
        for _ in range(num_iterations):

            # Send and receive calibration waveform
            calibration_transmitter.transmit(calibration_signal)

            _ = self.transmit()
            self.trigger()
            _ = self.receive()

            propagated_signal = calibration_receiver.receive().signal

            # Infer the implicit delay by estimating the sample index of the propagated dirac
            propagated_signals.append(propagated_signal)
            propagated_dirac_indices = np.argmax(abs(propagated_signal.samples[0, :]))

            # Wait the configured amount of time between iterations
            sleep(wait)

        # Un-register operators
        self.transmitters.remove(calibration_transmitter)
        self.receivers.remove(calibration_receiver)

        # Compute calibration delay
        # This is just a ML estimation
        mean_dirac_index = np.mean(propagated_dirac_indices)
        calibration_delay = (dirac_index - mean_dirac_index) / propagated_signal.sampling_rate

        # Dump calibration to pickle savefile
        with open(calibration_file, "wb") as file_stream:
            dump(calibration_delay, file_stream)

        # Save register calibration delay internally
        self.__calibration_delay = calibration_delay

        # Visualize the results
        fig, axes = plt.subplots(2, 1)
        fig.suptitle("Device Delay Calibration")

        axes[0].plot(calibration_signal.timestamps, abs(calibration_signal.samples[0, :]))
        for sig in propagated_signals:
            axes[1].plot(sig.timestamps, abs(sig.samples[0, :]), color="blue")
        axes[1].axvline(x=(dirac_index / sampling_rate - calibration_delay), color="red")

    def load_calibration(self, calibration_file: str) -> None:
        """Recall a dumped calibration from the filesystem.

        Args:
            calibration_file (str):
                Location of the calibration dump.
        """

        with open(calibration_file, "rb") as file_stream:
            self.__calibration_delay = load(file_stream)


PhysicalDeviceType = TypeVar("PhysicalDeviceType", bound=PhysicalDevice)
"""Type of phyiscal device."""
