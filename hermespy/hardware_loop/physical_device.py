# -*- coding: utf-8 -*-
"""
================
Physical Devices
================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Type, TypeVar

import numpy as np
from h5py import File, Group
from scipy.signal import butter, sosfilt

from hermespy.core import (
    Device,
    DeviceInput,
    DeviceReception,
    DeviceTransmission,
    Factory,
    HDFSerializable,
    ProcessedDeviceInput,
    Serializable,
    Signal,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


PDT = TypeVar("PDT", bound="PhysicalDevice")
"""Type variable for :class:`PhysicalDevice` and its subclasses."""


CT = TypeVar("CT", bound="Calibration")
"""Type variable for :class:`Calibration` and its subclasses."""


class PhysicalDevice(Device, ABC):
    """Base representing any device controlling real hardware."""

    __max_receive_delay: float
    __adaptive_sampling: bool
    __lowpass_filter: bool
    __lowpass_bandwidth: float
    __recent_upload: Signal | None
    __noise_power: np.ndarray | None
    __leakage_calibration: LeakageCalibrationBase
    __delay_calibration: DelayCalibrationBase

    def __init__(
        self,
        max_receive_delay: float = 0.0,
        noise_power: np.ndarray | None = None,
        leakage_calibration: LeakageCalibrationBase | None = None,
        delay_calibration: DelayCalibrationBase | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            max_receive_delay (float, optional):
                Maximum expected delay during siganl reception in seconds.
                Zero by default.

            noise_power (np.ndarray, optional):
                Assumed power of the hardware noise at the device's receive chains.

            leakage_calibration (LeakageCalibrationBase, optional):
                The leakage calibration routine to apply to received signals.
                If not provided, defaults to the :class:`NoLeakageCalibration` stub routine.

            delay_calibration (DelayCalibrationBase, optional):
                The delay calibration routine to apply to received signals.
                If not provided, defaults to the :class:`NoDelayCalibration` stub routine.

            *args:
                :class:`Device` base class initialization parameters.

            \**kwargs:
                :class:`Device` base class initialization parameters.
        """

        # Initialize base class
        Device.__init__(self, *args, **kwargs)

        # Initialize class attributes
        self.max_receive_delay = max_receive_delay
        self.__adaptive_sampling = False
        self.__lowpass_filter = False
        self.__lowpass_bandwidth = 0.0
        self.__recent_upload = None
        self.__noise_power = noise_power

        self.__leakage_calibration = NoLeakageCalibration()
        if leakage_calibration is None:
            self.__leakage_calibration.device = self
        else:
            self.leakage_calibration = leakage_calibration

        self.__delay_calibration = NoDelayCalibration()
        if delay_calibration is None:
            self.__delay_calibration.device = self
        else:
            self.delay_calibration = delay_calibration

    @abstractmethod
    def trigger(self) -> None:
        """Trigger the device."""
        ...  # pragma no cover

    @property
    def leakage_calibration(self) -> LeakageCalibrationBase:
        """Leakage calibration routine to apply to received signals.

        Returns:
            Handle to the calibration routine.
        """

        return self.__leakage_calibration

    @leakage_calibration.setter
    def leakage_calibration(self, value: LeakageCalibrationBase) -> None:
        if self.__leakage_calibration is value:
            return

        self.__leakage_calibration = value

        if self.__leakage_calibration.device is not self:
            self.__leakage_calibration.device = self

    @property
    def delay_calibration(self) -> DelayCalibrationBase:
        """Delay calibration routine to apply to received signals.

        Returns:
            Handle to the delay calibration routine.
        """

        return self.__delay_calibration

    @delay_calibration.setter
    def delay_calibration(self, value: DelayCalibrationBase) -> None:
        if self.__delay_calibration is value:
            return

        self.__delay_calibration = value
        self.__delay_calibration.device = self

    def save_calibration(self, path: str) -> None:
        """Save the calibration file to the hard drive.

        Args:

            path (str):
                Path under which the file should be stored.
        """

        file = File(path, "w")

        # Write leakage calibration to file
        file.attrs[LeakageCalibrationBase.hdf_group_name] = self.leakage_calibration.yaml_tag
        self.leakage_calibration.to_HDF(file.create_group("leakage_calibration"))

        # Write delay calibration to file
        file.attrs[DelayCalibrationBase.hdf_group_name] = self.delay_calibration.yaml_tag
        self.delay_calibration.to_HDF(file.create_group("delay_calibration"))

        # Close file handle
        file.close()

    @staticmethod
    def __calibration_from_hdf(file: File, factory: Factory, calibration: Type[CT]) -> CT | None:
        """Load a calibration object from an HDF file.

        Subroutine of :meth:`PhysicalDevice.load_calibration`.

        Args:

            file (File):
                File handle of the HDF file.

            factory (Factory):
                Factory object to create the calibration object.

            calibration (Type[CT]):
                Type of the calibration object to load.

        Returns:

            The loaded calibration object.
            `None` if the calibration object is not present in the HDF file.
        """

        group_name = calibration.hdf_group_name

        # If the tag isn't specified in the attribute list, the calibration object is not present in the file
        if group_name not in file.attrs:
            return None

        calibration_class: Type[CT] = factory.tag_registry[file.attrs.get(group_name)]  # type: ignore
        return calibration_class.from_HDF(file[group_name])

    def load_calibration(self, path: str) -> None:
        """Load the calibration file from the hard drive.

        Args:

            path (str):
                Path from which to load the calibration file.
        """

        file = File(path, "r")
        factory = Factory()

        # Load leakage calibration if available in save file
        leakage_calibration = self.__calibration_from_hdf(file, factory, LeakageCalibrationBase)  # type: ignore
        if leakage_calibration is not None:
            self.leakage_calibration = leakage_calibration

        # Load delay calibration if available in save file
        delay_calibration = self.__calibration_from_hdf(file, factory, DelayCalibrationBase)  # type: ignore
        if delay_calibration is not None:
            self.delay_calibration = delay_calibration

        file.close()

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

    @property
    def noise_power(self) -> np.ndarray | None:
        """Assumed hardware noise power at the device's receive chains.

        Returns:
            Numpy array indicating the noise power at each receive chain.
            `None` if unknown or not specified.

        Raises:

            ValueError: If any power is negative.
        """

        return self.__noise_power

    @noise_power.setter
    def noise_power(self, value: np.ndarray | None) -> None:
        if value is None:
            self.__noise_power = None
            return

        if np.any(value < 0.0):
            raise ValueError("Assumed hardware noise powers must be non-negative")

        self.__noise_power = value

    def estimate_noise_power(self, num_samples: int = 1000, cache: bool = True) -> np.ndarray:
        """Estimate the power of the noise floor within the hardware.

        Receives `num_samples` without any transmissions and estimates the power over them.
        Note that any interference in this case will be interpreted as noise.

        Args:

            num_samples (int, optional):
                Number of received samples.
                1000 by default.

            cache (bool, optional):
                If enabled, the :meth:`PhysicalDevice.noise_power` property will be updated.

        Returns:
            np.ndarray:
                Estimated noise power at each respective receive channel,
                i.e. the variance of the unbiased received samples.
        """

        silent_signal = Signal.Create(
            np.zeros((self.antennas.num_transmit_ports, num_samples)),
            self.sampling_rate,
            self.carrier_frequency,
        )
        noise_signal = self.trigger_direct(silent_signal)

        noise_power = noise_signal.power

        if cache:
            self.noise_power = noise_power

        return noise_power

    def _upload(self, signal: Signal) -> Signal:
        """Upload samples to be transmitted to the represented hardware.

        Args:

            signal (Signal):
                The samples to be uploaded.

        Returns: The actually uploaded samples, including quantization scaling.
        """

        # The default routine is a stub
        return signal

    def transmit(self, clear_cache: bool = True) -> DeviceTransmission:
        # If adaptive sampling is disabled, resort to the default transmission routine
        if not self.adaptive_sampling or self.transmitters.num_operators < 1:
            device_transmission = Device.transmit(self, clear_cache)

        else:
            operator_transmissions = self.transmit_operators()

            try:
                output = self.generate_output(operator_transmissions, False)

            except RuntimeError:
                raise RuntimeError("Resampling is required for adaptive sampling but not allowed")

            # Adaptive sampling rate will simply assume the device's sampling rate
            mixed_signal = output.mixed_signal
            mixed_signal.carrier_frequency = self.carrier_frequency
            mixed_signal.sampling_rate = self.sampling_rate

            # Generate the device transmission
            device_transmission = DeviceTransmission(operator_transmissions, mixed_signal)

        # Upload the samples
        uploaded_samples = self._upload(device_transmission.mixed_signal)
        self.__recent_upload = uploaded_samples

        # Return transmission
        return device_transmission

    def _download(self) -> Signal:
        """Download received samples from the represented hardware.

        Returns:
            A signal model of the downloaded samples.
        """

        # This method is a stub by default
        raise NotImplementedError("The default physical device does not support downloads")

    def trigger_direct(self, signal: Signal, calibrate: bool = True) -> Signal:
        """Trigger a direct transmission and reception.

        Bypasses the operator abstraction layer and directly transmits the given signal.

        Args:

            signal (Signal):
                The signal to be transmitted.

            calibrate (bool, optional):
                If enabled, the signal will be calibrated using the device's leakage calibration.
                Enabled by default.

        Returns: The received signal.

        Raises:

            ValueError: If the signal's carrier frequency or sampling rate does not match the device's.
        """

        if signal.carrier_frequency != self.carrier_frequency:
            raise ValueError(
                f"Carrier frequency of the transmitted signal does not match the device's carrier frequency ({signal.carrier_frequency} != {self.carrier_frequency})"
            )

        if signal.sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Sampling rate of the transmitted signal does not match the device's sampling rate ({signal.sampling_rate} != {self.sampling_rate})"
            )

        # Upload signal to the device's memory
        uploaded_samples = self._upload(signal)

        # Trigger transmission / reception
        self.trigger()

        # Download signal from device's memory
        received_signal = self._download()

        # Apply calibration correction
        corrected_signal = (
            received_signal
            if not calibrate or self.leakage_calibration is None
            else self.leakage_calibration.remove_leakage(
                uploaded_samples, received_signal, self.delay_calibration.delay
            )
        )

        return corrected_signal

    def process_input(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | None = None,
        cache: bool = True,
    ) -> ProcessedDeviceInput:
        # Physical devices are able to infer their impinging signals by downloading them directly
        if impinging_signals is None:
            # Download signal samples
            filtered_signal = self._download().copy()

            # Apply a lowpass filter if the respective physical device flag is enabled
            if self.lowpass_filter:
                if self.lowpass_bandwidth <= 0.0:
                    filter_cutoff = 0.25 * self.sampling_rate

                else:
                    filter_cutoff = 0.5 * self.lowpass_bandwidth

                filter = butter(5, filter_cutoff, output="sos", fs=self.sampling_rate)
                filtered_signal.set_samples(sosfilt(filter, filtered_signal[:, :], axis=1))

            _impinging_signals = filtered_signal

        elif isinstance(impinging_signals, DeviceInput):
            _impinging_signals = impinging_signals.impinging_signals[0]

        elif isinstance(impinging_signals, Sequence):
            _impinging_signals = impinging_signals[0]

        else:
            _impinging_signals = impinging_signals

        if _impinging_signals.num_streams != self.num_receive_ports:
            raise ValueError(
                "Number of received signal streams does not match number of configured antennas"
            )

        if _impinging_signals.sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Received signal sampling rate ({_impinging_signals.sampling_rate}) does not match device sampling rate({self.sampling_rate})"
            )

        # Apply calibration correction
        transmitted_signal = (
            self.__recent_upload
            if self.__recent_upload is not None
            else Signal.Empty(
                _impinging_signals.sampling_rate,
                _impinging_signals.num_streams,
                carrier_frequency=_impinging_signals.carrier_frequency,
            )
        )
        corrected_signal = self.leakage_calibration.remove_leakage(
            transmitted_signal, _impinging_signals, self.delay_calibration.delay
        )

        # Defer to the default device input processing routine
        return Device.process_input(self, corrected_signal, cache)

    def receive(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | None = None,
        cache: bool = True,
    ) -> DeviceReception:
        # Process input
        processed_input = self.process_input(impinging_signals)

        # Generate receptions
        operator_receptions = self.receive_operators(processed_input.operator_inputs, cache=cache)

        # Return reception
        return DeviceReception.From_ProcessedDeviceInput(processed_input, operator_receptions)


class Calibration(ABC, HDFSerializable, Serializable):
    """Abstract basse class of all calibration classes."""

    hdf_group_name = "calibration"
    """Group name of the calibration in the HDF save file."""

    __device: PhysicalDevice | None

    def __init__(self, device: PhysicalDevice | None = None) -> None:
        # Initialize class attributes
        self.__device = None
        self.device = device

    @classmethod
    @abstractmethod
    def _configure_slot(
        cls: Type[CT], device: PhysicalDevice, value: CT | None
    ) -> None: ...  # pragma: no cover

    @property
    def device(self) -> PhysicalDevice | None:
        return self.__device

    @device.setter
    def device(self, value: PhysicalDevice | None) -> None:
        # Do nothing if the calibrateable is already set to the new value
        if self.__device is value:
            return

        # Reset old configured device to default
        if self.__device is not None:
            self._configure_slot(self.__device, None)

        # Save new device and configure it to this calibration
        self.__device = value
        if self.__device is not None:
            self._configure_slot(self.__device, self)

    def save(self, path: str) -> None:
        """Save the calibration file to the hard drive.

        Args:

            path (str):
                Path under which the file should be stored.

        Raises:
        """

        file = File(path, "a")

        # Create dataset for the calibrateable
        dataset = file.create_group(self.hdf_group_name)

        # Save state to dataset
        self.to_HDF(dataset)

        # Close file handle
        file.close()

    @classmethod
    def Load(cls: Type[CT], source: str | Group) -> CT:
        if isinstance(source, str):
            file = File(source, "r")

            # Load dataset for the calibrateable
            group = file[cls.hdf_group_name]

        else:
            group = source

        # Recall and return the calibration from the HDF dataset
        calibration = cls.from_HDF(group)

        # Close file handle
        if isinstance(source, str):
            file.close()

        return calibration


class DelayCalibrationBase(Calibration, ABC):
    """Abstract base class for all delay calibration classes."""

    hdf_group_name = "delay_calibration"
    """Group name of the delay calibration in the HDF save file."""

    @classmethod
    def _configure_slot(
        cls: Type[DelayCalibrationBase], device: PhysicalDevice, value: DelayCalibrationBase | None
    ) -> None:
        device.delay_calibration = NoDelayCalibration() if value is None else value

    @property
    @abstractmethod
    def delay(self) -> float:
        """Expected transmit-receive delay in seconds."""
        ...  # pragma: no cover

    def correct_transmit_delay(self, signal: Signal) -> Signal:
        """Apply the delay calibration to a transmitted signal.

        Args:

            signal (Signal):
                The signal to be corrected.

        Returns:
            The corrected signal.
        """

        # Do nothing for positive delays
        if self.delay >= 0:
            return signal

        # Prepend zeros to the signal to account for negative delays
        delay_in_samples = round(-self.delay * signal.sampling_rate)
        signal.set_samples(
            np.concatenate(
                (np.zeros((signal.num_streams, delay_in_samples), dtype=np.complex_), signal[:, :]),
                axis=1,
            )
        )

        return signal

    def correct_receive_delay(self, signal: Signal) -> Signal:
        """Apply the delay calibration to a received signal.

        Args:

            signal (Signal):
                The signal to be corrected.

        Returns:
            The corrected signal.
        """

        # Do nothing for zero delays
        if self.delay <= 0:
            return signal

        # Remove samples from the signal to account for positive delays
        delay_in_samples = round(self.delay * signal.sampling_rate)
        signal.set_samples(signal[:, delay_in_samples:])

        return signal


class NoDelayCalibration(DelayCalibrationBase, Serializable):
    """No delay calibration."""

    yaml_tag = "NoDelayCalibration"

    @property
    def delay(self) -> float:
        return 0.0

    def correct_transmit_delay(self, signal: Signal) -> Signal:
        return signal

    def correct_receive_delay(self, signal: Signal) -> Signal:
        return signal

    def to_HDF(self, group: Group) -> None:
        return

    @classmethod
    def from_HDF(cls: Type[NoDelayCalibration], group: Group) -> NoDelayCalibration:
        return cls()


class LeakageCalibrationBase(Calibration):
    """Abstract base class for all leakage calibration classes."""

    hdf_group_name = "leakage_calibration"
    """Group name of the leakage calibration in the HDF save file."""

    @classmethod
    def _configure_slot(
        cls: Type[LeakageCalibrationBase],
        device: PhysicalDevice,
        value: LeakageCalibrationBase | None,
    ) -> None:
        device.leakage_calibration = NoLeakageCalibration() if value is None else value

    @abstractmethod
    def remove_leakage(
        self, transmitted_signal: Signal, received_signal: Signal, delay_correction: float = 0.0
    ) -> Signal:
        """Remove leakage from a received signal.

        Args:

            transmitted_signal (Signal):
                The transmitted signal.

            recveived_signal (Signal):
                The received signal from which to remove the leakage.

            delay_correction (float, optional):
                Delay correction applied by the device during reception in seconds.
                Assumed zero by default.

        Returns:
            The signal with removed leakage.
        """
        ...  # pragma: no cover


class NoLeakageCalibration(LeakageCalibrationBase, Serializable):
    """No leakage calibration."""

    yaml_tag = "NoLeakageCalibration"

    def remove_leakage(
        self, transmitted_signal: Signal, received_signal: Signal, delay_correction: float = 0.0
    ) -> Signal:
        return received_signal

    def to_HDF(self, group: Group) -> None:
        return

    @classmethod
    def from_HDF(cls: Type[NoLeakageCalibration], group: Group) -> NoLeakageCalibration:
        return cls()
