# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import chain
from typing import List, Set, Type

import numpy as np
from h5py import Group

from hermespy.core import (
    AntennaArrayState,
    Device,
    DeviceInput,
    DeviceOutput,
    DeviceReception,
    DeviceTransmission,
    HDFSerializable,
    ProcessedDeviceInput,
    RandomNode,
    Transformation,
    Transmission,
    Reception,
    Scenario,
    Serializable,
    Signal,
    Receiver,
    register,
)
from .animation import Moveable, StaticTrajectory, Trajectory, TrajectorySample
from .antennas import SimulatedAntennaArray, SimulatedIdealAntenna, SimulatedUniformArray
from .noise import NoiseLevel, NoiseModel, SNR, NoiseRealization, AWGN
from .rf_chain.rf_chain import RfChain
from .isolation import Isolation, PerfectIsolation
from .coupling import Coupling, PerfectCoupling

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TriggerRealization(HDFSerializable):
    """Realization of a trigger model.

    A trigger realization will contain the offset between a drop start and the waveform frames contained within the drop.
    """

    __num_offset_samples: int
    __sampling_rate: float

    def __init__(self, num_offset_samples: int, sampling_rate: float) -> None:
        """
        Args:

            num_offset_samples (int):
                Number of discrete samples between drop start and frame start

            sampling_rate (float):
                Sampling rate at which the realization was generated in Hz.


        Raises:
            ValueError: For `num_offset_samples` smaller than zero.
            ValueError: For a `sampling_rate` smaller or equal to zero.
        """

        if num_offset_samples < 0:
            raise ValueError(
                f"Number of offset samples must be non-negative (not {num_offset_samples})"
            )

        if sampling_rate <= 0.0:
            raise ValueError(
                f"Sampling rate must be greater or equal to zero (not {sampling_rate})"
            )

        self.__num_offset_samples = num_offset_samples
        self.__sampling_rate = sampling_rate

    @property
    def num_offset_samples(self) -> int:
        """Number of discrete samples between drop start and frame start."""

        return self.__num_offset_samples

    @property
    def sampling_rate(self) -> float:
        """Sampling rate at which the realization was generated in Hz."""

        return self.__sampling_rate

    @property
    def trigger_delay(self) -> float:
        """Time between drop start and frame start in seconds."""

        return self.num_offset_samples / self.sampling_rate

    def compute_num_offset_samples(self, sampling_rate: float) -> int:
        """Compute the number of realized offset samples for a custom sampling rate.

        The result is rounded to the nearest smaller integer.

        Args:

            sampling_rate (sampling_rate, float):
                Sampling rate of intereset in Hz.

        Returns:
            The number of trigger offset samples.

        Raises:
            ValueError: For a `sampling_rate` smaller or equal to zero.
        """

        if sampling_rate <= 0.0:
            raise ValueError(
                f"Sampling rate must be greater or equal to zero (not {sampling_rate})"
            )

        if sampling_rate == self.sampling_rate:
            return self.num_offset_samples

        num_offset_samples = int(self.num_offset_samples * sampling_rate / self.sampling_rate)
        return num_offset_samples

    def to_HDF(self, group: Group) -> None:
        group.attrs["num_offset_samples"] = self.num_offset_samples
        group.attrs["sampling_rate"] = self.sampling_rate

    @classmethod
    def from_HDF(cls: Type[TriggerRealization], group: Group) -> TriggerRealization:
        num_offset_samples = group.attrs.get("num_offset_samples", 0)
        sampling_rate = group.attrs.get("sampling_rate", 1.0)

        return cls(num_offset_samples, sampling_rate)


class TriggerModel(ABC, RandomNode):
    """Base class for all trigger models."""

    __devices: Set[SimulatedDevice]

    def __init__(self) -> None:
        # Initialize base classes
        RandomNode.__init__(self)

        # Initialize class attributes
        self.__devices = set()

    def add_device(self, device: SimulatedDevice) -> None:
        """Add a new device to be controlled by this trigger.

        Args:

            device (SimulatedDevice):
                The device to be controlled by this trigger.
        """

        if device.trigger_model is not self:
            device.trigger_model = self

        self.__devices.add(device)

    def remove_device(self, device: SimulatedDevice) -> None:
        """Remove a device from being controlled by this trigger.

        Args:

            device (SimulatedDevice): The device to be removed.
        """

        self.__devices.discard(device)

    @property
    def num_devices(self) -> int:
        """Number of devices controlled by this trigger."""

        return len(self.__devices)

    @property
    def devices(self) -> Set[SimulatedDevice]:
        """Set of devices controlled by this trigger."""

        return self.__devices

    @abstractmethod
    def realize(self, rng: np.random.Generator | None = None) -> TriggerRealization:
        """Realize a triggering of all controlled devices.

        Args:

            rng (np.random.Generator, optional):
                Random number generator used to realize this trigger model.
                If not specified, the object's internal generator will be queried.

        Returns: Realization of the trigger model.
        """
        ...  # pragma: no cover


class StaticTrigger(TriggerModel, Serializable):
    """Model of a trigger that's always perfectly synchronous with the drop start."""

    yaml_tag = "StaticTrigger"

    def realize(self, rng: np.random.Generator | None = None) -> TriggerRealization:
        if self.num_devices < 1:
            sampling_rate = 1.0

        else:
            sampling_rate = list(self.devices)[0].sampling_rate

        # Static triggers will always consider a perfect trigger at the drop start
        return TriggerRealization(0, sampling_rate)


class SampleOffsetTrigger(TriggerModel, Serializable):
    """Model of a trigger that generates a constant offset of samples between drop start and frame start"""

    yaml_tag = "TimeOffsetTrigger"
    __num_offset_samples: int

    def __init__(self, num_offset_samples: int) -> None:
        """
        Args:

            num_offset_samples (int):
                Number of discrete samples between drop start and frame start.
        """

        # Initialize base classes
        TriggerModel.__init__(self)
        Serializable.__init__(self)

        # Initialize class attributes
        self.num_offset_samples = num_offset_samples

    @property
    def num_offset_samples(self) -> int:
        """Number of discrete samples between drop start and frame start.

        Returns: Number of samples
        """

        return self.__num_offset_samples

    @num_offset_samples.setter
    def num_offset_samples(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"Synchronization offset must be non-negatve (not {value})")

        self.__num_offset_samples = value

    def realize(self, rng: np.random.Generator | None = None) -> TriggerRealization:
        if self.num_devices < 1:
            raise RuntimeError(
                "Realizing a static trigger requires the trigger to control at least one device"
            )

        sampling_rate = list(self.devices)[0].sampling_rate
        return TriggerRealization(self.num_offset_samples, sampling_rate)


class TimeOffsetTrigger(TriggerModel, Serializable):
    """Model of a trigger that generates a constant time offset between drop start and frame start.

    Note that the offset is rounded to the nearest smaller integer number of samples,
    depending on the sampling rate of the first controlled device's sampling rate.
    """

    yaml_tag = "TimeOffsetTrigger"
    __offset: float

    def __init__(self, offset: float) -> None:
        """
        Args:

            offset (float):
                Offset between drop start and frame start in seconds.
        """

        # Initialize base classes
        TriggerModel.__init__(self)
        Serializable.__init__(self)

        # Initialize class attributes
        self.offset = offset

    @property
    def offset(self) -> float:
        """Offset between drop start and frame start in seconds."""

        return self.__offset

    @offset.setter
    def offset(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Synchronization offset must be non-negatve (not {value})")

        self.__offset = value

    def realize(self, rng: np.random.Generator | None = None) -> TriggerRealization:
        if self.num_devices < 1:
            raise RuntimeError(
                "Realizing a static trigger requires the trigger to control at least one device"
            )

        sampling_rate = list(self.devices)[0].sampling_rate
        num_offset_samples = int(self.offset * sampling_rate)

        return TriggerRealization(num_offset_samples, sampling_rate)


class RandomTrigger(TriggerModel, Serializable):
    """Model of a trigger that generates a random offset between drop start and frame start"""

    yaml_tag = "RandomTrigger"

    def realize(self, rng: np.random.Generator | None = None) -> TriggerRealization:
        if self.num_devices < 1:
            raise RuntimeError(
                "Realizing a random trigger requires the trigger to control at least one device"
            )

        devices = list(self.devices)
        sampling_rate = devices[0].sampling_rate
        max_frame_duration = devices[0].max_frame_duration

        for device in devices[1:]:
            # Make sure all devices match in their sampling rate
            if device.sampling_rate != sampling_rate:
                raise RuntimeError(
                    "Random trigger groups only support devices of identical sampling rate"
                )

            # Look for the maximum frame duration, which will determine the unform distribution to be realized
            max_frame_duration = max(max_frame_duration, device.max_frame_duration)

        max_trigger_delay = int(max_frame_duration * sampling_rate)

        if max_trigger_delay == 0:
            return TriggerRealization(0, sampling_rate)

        # Select the proper random number generator
        _rng = self._rng if rng is None else rng

        trigger_delay = _rng.integers(0, max_trigger_delay)
        return TriggerRealization(trigger_delay, sampling_rate)


class SimulatedDeviceOutput(DeviceOutput):
    """Information transmitted by a simulated device"""

    __emerging_signals: Sequence[Signal]
    __trigger_realization: TriggerRealization

    def __init__(
        self,
        emerging_signals: Signal | Sequence[Signal],
        trigger_realization: TriggerRealization,
        sampling_rate: float,
        num_antennas: int,
        carrier_frequency: float,
    ) -> None:
        """
        Args:

            emerging_signals (Signal | Sequence[Signal]):
                Signal models emerging from the device.

            trigger_realization (TriggerRealization):
                Trigger realization modeling the time delay between a drop start and frame start.

            sampling_rate (float):
                Device sampling rate in Hz during the transmission.

            num_antennas (int):
                Number of transmitting device antennas.

            carrier_frequency (float):
                Device carrier frequency in Hz.

        Raises:

            ValueError: If `sampling_rate` is greater or equal to zero.
            ValueError: If `num_antennas` is smaller than one.
        """

        _emerging_signals = (
            [emerging_signals] if isinstance(emerging_signals, Signal) else emerging_signals
        )
        superimposed_signal = Signal.Empty(
            sampling_rate, num_antennas, carrier_frequency=carrier_frequency
        )

        # Assert emerging signal's validity and superimpose the signals
        for signal in _emerging_signals:
            if signal.sampling_rate != sampling_rate:
                raise ValueError(
                    f"Emerging signal has unexpected sampling rate ({signal.sampling_rate} instad of {sampling_rate})"
                )

            if signal.num_streams != num_antennas:
                raise ValueError(
                    f"Emerging signal has unexpected number of transmit antennas ({signal.num_streams} instead of {num_antennas})"
                )

            if signal.carrier_frequency != carrier_frequency:
                raise ValueError(
                    f"Emerging signal has unexpected carrier frequency ({signal.carrier_frequency} instead of {carrier_frequency})"
                )

            superimposed_signal.superimpose(signal)

        # Initialize attributes
        self.__trigger_realization = trigger_realization
        self.__emerging_signals = _emerging_signals

        # Initialize base class
        DeviceOutput.__init__(self, superimposed_signal)

    @classmethod
    def From_DeviceOutput(
        cls: Type[SimulatedDeviceOutput],
        device_output: DeviceOutput,
        emerging_signals: Signal | Sequence[Signal],
        trigger_realization: TriggerRealization,
    ) -> SimulatedDeviceOutput:
        """Initialize a simulated device output from its base class.

        Args:

            device_output (DeviceOutput):
                Device output.

            emerging_signals (Union[Signal, List[Signal]]):
                Signal models emerging from the device.

            trigger_realization (TriggerRealization):
                Trigger realization modeling the time delay between a drop start and frame start.

        Returns: The initialized object.
        """

        return cls(
            emerging_signals,
            trigger_realization,
            device_output.sampling_rate,
            device_output.num_antennas,
            device_output.carrier_frequency,
        )

    @property
    def trigger_realization(self) -> TriggerRealization:
        """Trigger realization modeling the time delay between a drop start and frame start.

        Returns: Handle to the trigger realization.
        """

        return self.__trigger_realization

    @property
    def operator_separation(self) -> bool:
        """Operator separation enabled?

        Returns: Operator separation indicator.
        """

        return len(self.__emerging_signals) > 1

    @property
    def emerging_signals(self) -> Sequence[Signal]:
        return self.__emerging_signals

    @classmethod
    def from_HDF(cls: Type[SimulatedDeviceOutput], group: Group) -> SimulatedDeviceOutput:
        # Recall base class
        device_output = DeviceOutput.from_HDF(group)

        # Recall emerging signals
        num_emerging_signals = group.attrs.get("num_emerging_signals", 0)
        emerging_signals = [
            Signal.from_HDF(group[f"emerging_signal_{s:02d}"]) for s in range(num_emerging_signals)
        ]

        # Recall trigger realization
        trigger_realization = TriggerRealization.from_HDF(group["trigger_realization"])

        # Initialize object
        return cls.From_DeviceOutput(device_output, emerging_signals, trigger_realization)

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        DeviceOutput.to_HDF(self, group)

        # Serialize emerging signals
        group.attrs["num_emerging_signals"] = self.num_emerging_signals
        for e, emerging_signal in enumerate(self.emerging_signals):
            emerging_signal.to_HDF(group.create_group(f"emerging_signal_{e:02d}"))

        # Serialize trigger realization
        self.trigger_realization.to_HDF(self._create_group(group, "trigger_realization"))


class SimulatedDeviceTransmission(DeviceTransmission, SimulatedDeviceOutput):
    """Information generated by transmitting over a simulated device."""

    def __init__(
        self,
        operator_transmissions: Sequence[Transmission],
        emerging_signals: Signal | Sequence[Signal],
        trigger_realization: TriggerRealization,
        sampling_rate: float,
        num_antennas: int,
        carrier_frequency: float,
    ) -> None:
        """
        Args:

            operator_transmissions (Sequence[Transmission]):
                Information generated by transmitting over transmit operators.

            emerging_signals (Signal | Sequence[Signal]):
                Signal models emerging from the device.

            trigger_realization (TriggerRealization):
                Trigger realization modeling the time delay between a drop start and frame start.

            sampling_rate (float):
                Device sampling rate in Hz during the transmission.

            num_antennas (int):
                Number of transmitting device antennas.

            carrier_frequency (float):
                Device carrier frequency in Hz.

        Raises:

            ValueError: If `sampling_rate` is greater or equal to zero.
            ValueError: If `num_antennas` is smaller than one.
        """

        # Initialize base classes
        SimulatedDeviceOutput.__init__(
            self,
            emerging_signals,
            trigger_realization,
            sampling_rate,
            num_antennas,
            carrier_frequency,
        )
        DeviceTransmission.__init__(self, operator_transmissions, SimulatedDeviceOutput.mixed_signal.fget(self))  # type: ignore

    @classmethod
    def From_SimulatedDeviceOutput(
        cls: Type[SimulatedDeviceTransmission],
        output: SimulatedDeviceOutput,
        operator_transmissions: Sequence[Transmission],
    ) -> SimulatedDeviceTransmission:
        return cls(
            operator_transmissions,
            output.emerging_signals,
            output.trigger_realization,
            output.sampling_rate,
            output.num_antennas,
            output.carrier_frequency,
        )

    @classmethod
    def from_HDF(
        cls: Type[SimulatedDeviceTransmission], group: Group
    ) -> SimulatedDeviceTransmission:
        # Recover base classes
        device_transmission = DeviceTransmission.from_HDF(group)
        devic_output = SimulatedDeviceOutput.from_HDF(group)

        # Initialize class from device output and operator transmissions
        return cls.From_SimulatedDeviceOutput(
            devic_output, device_transmission.operator_transmissions
        )

    def to_HDF(self, group: Group) -> None:
        DeviceTransmission.to_HDF(self, group)
        SimulatedDeviceOutput.to_HDF(self, group)


class SimulatedDeviceReceiveRealization(object):
    """Realization of a simulated device reception random process."""

    __noise_realization: NoiseRealization

    def __init__(self, noise_realization: NoiseRealization) -> None:
        """
        Args:

            noise_realization (Sequence[NoiseRealization]):
                Noise realizations for each receive operator.
        """

        self.__noise_realization = noise_realization

    @property
    def noise_realization(self) -> NoiseRealization:
        """Receive operator noise realizations.

        Returns: Sequence of noise realizations corresponding to the number of registerd receive operators.
        """

        return self.__noise_realization


class ProcessedSimulatedDeviceInput(SimulatedDeviceReceiveRealization, ProcessedDeviceInput):
    """Information generated by receiving over a simulated device."""

    __leaking_signal: Signal | None
    __baseband_signal: Signal
    __operator_separation: bool
    __trigger_realization: TriggerRealization

    def __init__(
        self,
        impinging_signals: Sequence[Signal | Sequence[Signal]],
        leaking_signal: Signal,
        baseband_signal: Signal,
        operator_separation: bool,
        operator_inputs: Sequence[Signal],
        noise_realization: NoiseRealization,
        trigger_realization: TriggerRealization,
    ) -> None:
        """
        Args:

            impinging_signals (Sequence[Sequence[Signal] | Signal]):
                Numpy vector containing lists of signals impinging onto the device.

            leaking_signal (Signal):
                Signal leaking from transmit to receive chains.

            baseband_signal (Signal):
                Baseband signal model from which the operator inputs are generated.

            operator_separation (bool):
                Is the operator separation flag enabled?

            operator_inputs (Sequence[Signal]):
                Information cached by the device operators.

            noise_realization (NoiseRealization):
                Realization of the device's noise model.

            trigger_realization (TriggerRealization):
                Trigger realization modeling the time delay between a drop start and frame start.

        Raises:

            ValueError: If `operator_separation` is enabled and `impinging_signals` contains sublists longer than one.
        """

        _impinging_signals: List[Signal] = []
        if len(impinging_signals) > 0:  # pragma: no cover
            if operator_separation and isinstance(impinging_signals[0], Sequence):  # type: ignore
                for signal_sequence in impinging_signals:
                    superposition = signal_sequence[0] if len(signal_sequence) > 0 else Signal.Empty(1.0, 1)  # type: ignore
                    for subsignal in signal_sequence[1:]:  # type: ignore
                        superposition.superimpose(subsignal)  # type: ignore

                    _impinging_signals.append(superposition)  # type: ignore

            elif not operator_separation and isinstance(impinging_signals[0], Signal):
                _impinging_signals = impinging_signals  # type: ignore

            else:
                raise ValueError("Invalid configuration")

        # Initialize base classes
        SimulatedDeviceReceiveRealization.__init__(self, noise_realization)
        ProcessedDeviceInput.__init__(self, _impinging_signals, operator_inputs)

        # Initialize attributes
        self.__leaking_signal = leaking_signal
        self.__baseband_signal = baseband_signal
        self.__operator_separation = operator_separation
        self.__trigger_realization = trigger_realization

    @property
    def leaking_signal(self) -> Signal | None:
        """Signal leaking from transmit to receive chains.

        Returns:
            Model if the leaking signal.
            `None` if no leakage was considered.
        """

        return self.__leaking_signal

    @property
    def baseband_signal(self) -> Signal:
        """Signal model from which operator inputs are generated."""

        return self.__baseband_signal

    @property
    def operator_separation(self) -> bool:
        """Operator separation flag.

        Returns: Boolean flag.
        """

        return self.__operator_separation

    @property
    def trigger_realization(self) -> TriggerRealization:
        """Trigger realization modeling the time delay between a drop start and frame start."""

        return self.__trigger_realization

    def to_HDF(self, group: Group) -> None:
        ProcessedDeviceInput.to_HDF(self, group)
        group.attrs["operator_separation"] = self.operator_separation
        if self.leaking_signal is not None:
            self.leaking_signal.to_HDF(self._create_group(group, "leaking_signal"))
        self.baseband_signal.to_HDF(self._create_group(group, "baseband_signal"))
        self.trigger_realization.to_HDF(self._create_group(group, "trigger_realization"))

    @classmethod
    def from_HDF(
        cls: Type[ProcessedSimulatedDeviceInput], group: Group
    ) -> ProcessedSimulatedDeviceInput:
        device_input = ProcessedDeviceInput.from_HDF(group)
        operator_separation = False  # group.attrs.get("operator_separation", False)
        leaking_signal = (
            Signal.from_HDF(group["leaking_signal"]) if "leaking_signal" in group else None
        )
        baseband_signal = Signal.from_HDF(group["baseband_signal"])
        noise_realization: NoiseRealization = None  # ToDo: Serialize noise realizations
        trigger_realization = TriggerRealization.from_HDF(group["trigger_realization"])

        return cls(
            device_input.impinging_signals,
            leaking_signal,
            baseband_signal,
            operator_separation,
            device_input.operator_inputs,
            noise_realization,
            trigger_realization,
        )


class SimulatedDeviceReception(ProcessedSimulatedDeviceInput, DeviceReception):
    """Information generated by receiving over a simulated device and its operators."""

    def __init__(
        self,
        impinging_signals: Sequence[Sequence[Signal]] | Sequence[Signal],
        leaking_signal: Signal,
        baseband_signal: Signal,
        operator_separation: bool,
        operator_inputs: Sequence[Signal],
        noise_realization: NoiseRealization,
        trigger_realization: TriggerRealization,
        operator_receptions: Sequence[Reception],
    ) -> None:
        """
        Args:

            impinging_signals (Sequence[Sequence[Signal]] | Sequence[Signal]):
                Sequences of signal models impinging onto the device from each linked device.

            leaking_signal (Signal):
                Signal leaking from transmit to receive chains.

            baseband_signal (Signal):
                Baseband signal model from which the operator inputs are generated.

            operator_separation (bool):
                Is the operator separation flag enabled?

            operator_inputs (Sequence[Signal]):
                Information cached by the device operators.

            noise_realization (NoiseRealization):
                Device noise realization.

            trigger_realization (TriggerRealization):
                Trigger realization modeling the time delay between a drop start and frame start.

            operator_receptions (Sequence[Reception]):
                Information inferred from receive operators.
        """

        ProcessedSimulatedDeviceInput.__init__(
            self,
            impinging_signals,
            leaking_signal,
            baseband_signal,
            operator_separation,
            operator_inputs,
            noise_realization,
            trigger_realization,
        )
        DeviceReception.__init__(self, self.impinging_signals, operator_inputs, operator_receptions)

    @classmethod
    def From_ProcessedSimulatedDeviceInput(
        cls: Type[SimulatedDeviceReception],
        device_input: ProcessedSimulatedDeviceInput,
        operator_receptions: Sequence[Reception],
    ) -> SimulatedDeviceReception:
        """Initialize a simulated device reception from a device input.

        Args:

            device (ProcessedSimulatedDeviceInput):
                The simulated device input.

            operator_receptions (Sequence[Reception]):
                Information received by device operators.

        Returns: The initialized object.
        """

        return cls(
            device_input.impinging_signals,
            device_input.leaking_signal,
            device_input.baseband_signal,
            device_input.operator_separation,
            device_input.operator_inputs,
            device_input.noise_realization,
            device_input.trigger_realization,
            operator_receptions,
        )

    def to_HDF(self, group: Group) -> None:
        ProcessedSimulatedDeviceInput.to_HDF(self, group)
        DeviceReception.to_HDF(self, group)

    @classmethod
    def from_HDF(cls: Type[SimulatedDeviceReception], group: Group) -> SimulatedDeviceReception:
        device_input = ProcessedSimulatedDeviceInput.from_HDF(group)
        device_reception = DeviceReception.from_HDF(group)

        return cls.From_ProcessedSimulatedDeviceInput(
            device_input, device_reception.operator_receptions
        )


class DeviceState(object):
    """Data container representing the immutable physical state of a simulated device during simulation drop generation.

    Generated by calling the :meth:`state<SimultedDevice.state>` property of a :class:`SimulatedDevice`.
    """

    __trajectory_sample: TrajectorySample
    __carrier_frequency: float
    __sampling_rate: float
    __antennas: AntennaArrayState

    def __init__(
        self,
        trajectory_sample: TrajectorySample,
        carrier_frequency: float,
        sampling_rate: float,
        antennas: AntennaArrayState,
    ) -> None:
        """
        Args:

            trajectory_sample (TrajectorySample):
                Position and orientation of the device in time and space.

            carrier_frequency (float):
                Carrier frequency of the device in Hz.

            sampling_rate (float):
                Sampling rate of the device in Hz.

            antennas (AntennaArrayState):
                State of the device's antenna array.
        """

        self.__trajectory_sample = trajectory_sample
        self.__carrier_frequency = carrier_frequency
        self.__sampling_rate = sampling_rate
        self.__antennas = antennas

    @property
    def position(self) -> np.ndarray:
        """Global position of the device in cartesian coordinates.

        Shorthand to :attr:`DeviceState.pose.translation`.
        """

        return self.__trajectory_sample.pose.translation

    @property
    def velocity(self) -> np.ndarray:
        """Global velocity of the device in m/s."""

        return self.__trajectory_sample.velocity

    @property
    def pose(self) -> Transformation:
        """Global pose of the device."""

        return self.__trajectory_sample.pose

    @property
    def carrier_frequency(self) -> float:
        """Carrier frequency of the device in Hz."""

        return self.__carrier_frequency

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the device in Hz."""

        return self.__sampling_rate

    @property
    def antennas(self) -> AntennaArrayState:
        """State of the device's antenna array."""

        return self.__antennas


class SimulatedDevice(Device, Moveable, Serializable):
    """Representation of an entity capable of emitting and receiving electromagnetic waves.

    A simulation scenario consists of a collection of devices,
    interconnected by a network of channel models.
    """

    yaml_tag = "SimulatedDevice"
    property_blacklist = {
        "num_antennas",
        "orientation",
        "random_mother",
        "scenario",
        "topology",
        "velocity",
        "wavelength",
    }
    serialized_attribute = {"rf_chain", "adc"}

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    __isolation: Isolation
    """Model of the device's transmit-receive isolations"""

    __coupling: Coupling
    """Model of the device's antenna array mutual coupling"""

    __trigger_model: TriggerModel
    """Model of the device's triggering behaviour"""

    __output: SimulatedDeviceOutput | None  # Most recent device output
    __input: ProcessedSimulatedDeviceInput | None  # Most recent device input
    __noise_level: NoiseLevel
    __noise_model: NoiseModel
    __scenario: Scenario | None  # Scenario this device is attached to
    __sampling_rate: float | None  # Sampling rate at which this device operate
    __carrier_frequency: float  # Center frequency of the mixed signal in rf-band
    __operator_separation: bool  # Operator separation flag
    __realization: (
        SimulatedDeviceReceiveRealization | None
    )  # Most recent device receive realization

    def __init__(
        self,
        scenario: Scenario | None = None,
        antennas: SimulatedAntennaArray | None = None,
        rf_chain: RfChain | None = None,
        isolation: Isolation | None = None,
        coupling: Coupling | None = None,
        trigger_model: TriggerModel | None = None,
        sampling_rate: float | None = None,
        carrier_frequency: float = 0.0,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        pose: Transformation | Trajectory | None = None,
        velocity: np.ndarray | None = None,
        power: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            antennas (SimulatedAntennaArray, optional):
                Antenna array model of the device.
                By default, a single ideal istropic antenna is assumed.

            rf_chain (RfChain, optional):
                Model of the device's radio frequency amplification chain.
                If not specified, a chain with ideal hardware models will be assumed.

            isolation (Isolation, optional):
                Model of the device's transmit-receive isolations.
                By default, perfect isolation is assumed.

            coupling (Coupling, optional):
                Model of the device's antenna array mutual coupling.
                By default, ideal coupling behaviour is assumed.

            trigger_model (TriggerModel, optional):
                The assumed trigger model.
                By default, a :class:`StaticTrigger` is assumed.

            sampling_rate (float, optional):
                Sampling rate at which this device operates.
                By default, the sampling rate of the first operator is assumed.

            carrier_frequency (float, optional):
                Center frequency of the mixed signal in rf-band in Hz.
                Zero by default.

            snr (float, optional):
                Signal-to-noise ratio of the device.
                By default, the device is assumed to be noiseless.

            snr_type (SNRType, optional):
                Type of the signal-to-noise ratio.
                By default, the signal-to-noise ratio is specified in terms of the power of the noise
                to the expected power of the received signal.

            pose (Transformation | Trajectory, optional):
                Position and orientation of the device in time and space.
                If a `Transformation` is provided, the device is assumed to be static.
                If not specified, the device is assumed to be at the origin with zero velocity.

            velocity (np.ndarray, optional):
                Initial velocity of the moveable in local coordinates.
                By default, the moveable is assumed to be resting.
                Only considered if `pose` is a `Transformation`, otherwise the velocity is assumed from the `Trajectory`.

            power (float, optional):
                Power of the device.
                Assumed to be 1.0 by default.

            seed (int, optional):
                Seed of the device's pseudo-random number generator.
        """

        # Init base classes
        _trajectory = pose if isinstance(pose, Trajectory) else StaticTrajectory(pose, velocity)
        Device.__init__(self, power, _trajectory.sample(0).pose, seed)
        Moveable.__init__(self, _trajectory)
        Serializable.__init__(self)

        # Initialize class attributes
        self.antennas = (
            SimulatedUniformArray(SimulatedIdealAntenna, 1.0, [1, 1, 1])
            if antennas is None
            else antennas
        )
        self.__scenario = None
        self.scenario = scenario
        self.rf_chain = RfChain() if rf_chain is None else rf_chain
        self.isolation = PerfectIsolation() if isolation is None else isolation
        self.coupling = PerfectCoupling() if coupling is None else coupling

        self.__trigger_model = StaticTrigger()
        self.__trigger_model.add_device(self)
        if trigger_model is not None:
            self.trigger_model = trigger_model

        self.noise_level = SNR(float("inf"), self) if noise_level is None else noise_level
        self.noise_model = AWGN() if noise_model is None else noise_model
        self.operator_separation = False
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.__input = None
        self.__output = None
        self.__realization = None

    @property
    def antennas(self) -> SimulatedAntennaArray:
        """Antenna array model of the simulated device."""

        return self.__antennas

    @antennas.setter
    def antennas(self, value: SimulatedAntennaArray) -> None:
        self.__antennas = value
        value.set_base(self)

    @property
    def scenario(self) -> Scenario | None:
        """Scenario this device is attached to.

        Returns:
            Handle to the scenario this device is attached to.
            `None` if the device is considered floating.
        """

        return self.__scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Set the scenario this device is attached to."""

        if self.__scenario is not scenario:
            # Pop the device from the old scenario
            if self.__scenario is not None:
                raise NotImplementedError()  # pragma: no cover

            self.__scenario = scenario
            self.random_mother = scenario

    @property
    def attached(self) -> bool:
        """Attachment state of this device.

        Returns:
            bool: `True` if the device is currently attached, `False` otherwise.
        """

        return self.__scenario is not None

    @register(first_impact="receive_devices", title="Device Noise Level")  # type: ignore[misc]
    @property
    def noise_level(self) -> NoiseLevel:
        """Level of the simulated hardware noise."""

        return self.__noise_level

    @noise_level.setter
    def noise_level(self, value: NoiseLevel) -> None:
        self.__noise_level = value

    @register(first_impact="receive_devices", title="Device Noise Model")  # type: ignore[misc]
    @property
    def noise_model(self) -> NoiseModel:
        """Model of the simulated hardware noise."""

        return self.__noise_model

    @noise_model.setter
    def noise_model(self, value: NoiseModel) -> None:
        self.__noise_model = value
        self.__noise_model.random_mother = self

    @register(first_impact="receive_devices", title="Device Noise Level")  # type: ignore[misc]
    @property
    def noise(self) -> float:
        """Shorthand property for accessing the noise level of the device.

        Raises:

            ValueError: For negative noise levels.
        """
        return self.noise_level.level

    @noise.setter
    def noise(self, value: float) -> None:
        self.noise_level.level = value

    @property
    def isolation(self) -> Isolation:
        """Model of the device's transmit-receive isolation.

        Returns: Handle to the isolation model.
        """

        return self.__isolation

    @isolation.setter
    def isolation(self, value: Isolation) -> None:
        self.__isolation = value
        value.device = self

    @property
    def coupling(self) -> Coupling:
        """Model of the device's antenna array mutual coupling behaviour.

        Returns: Handle to the coupling model.
        """

        return self.__coupling

    @coupling.setter
    def coupling(self, value: Coupling) -> None:
        self.__coupling = value
        value.device = self

    @property
    def trigger_model(self) -> TriggerModel:
        """The device's trigger model."""

        return self.__trigger_model

    @trigger_model.setter
    def trigger_model(self, value: TriggerModel) -> None:
        # Remove the self-reference from the old trigger model
        self.__trigger_model.remove_device(self)

        # Adopt the new trigger model and add a self-reference to its list of devices
        self.__trigger_model = value
        value.add_device(self)

    @property
    def sampling_rate(self) -> float:
        """Sampling rate at which the device's analog-to-digital converters operate.

        Returns:
            Sampling rate in Hz.
            If no operator has been specified and the sampling rate was not set,
            a sampling rate of :math:`1` Hz will be assumed by default.

        Raises:
            ValueError: If the sampling rate is not greater than zero.
        """

        if self.__sampling_rate is not None:
            return self.__sampling_rate

        sampling_rate = 0.0
        for operator in chain(self.transmitters, self.receivers):
            sampling_rate = max(sampling_rate, operator.sampling_rate)  # type: ignore

        return 1.0 if sampling_rate == 0.0 else sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float | None) -> None:
        if value is None:
            self.__sampling_rate = None
            return

        if value <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def carrier_frequency(self) -> float:
        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Carrier frequency must be greater or equal to zero")

        self.__carrier_frequency = value

    def state(self, timestamp: float = 0.0) -> DeviceState:
        """Immutable physical state of the device during simulation drop generation.

        Args:

            timestamp (float, optional):
                Global timestamp in seconds.
                By default, zero is assumed.

        Returns: Device state at the given timestamp.
        """

        trajectory_sample = self.trajectory.sample(timestamp)

        return DeviceState(
            trajectory_sample,
            self.carrier_frequency,
            self.sampling_rate,
            self.antennas.state(trajectory_sample.pose),
        )

    @property
    def operator_separation(self) -> bool:
        """Separate operators during signal modeling.

        Returns:

            Enabled flag.
        """

        return self.__operator_separation

    @operator_separation.setter
    def operator_separation(self, value: bool) -> None:
        self.__operator_separation = value

    def _simulate_output(self, signal: Signal) -> Signal:
        """Simulate a device output over the device's hardware model.

        Args:

            signal (Signal): Signal feeding into the hardware chain.

        Returns: Signal emerging from the hardware chain.
        """

        # Simulate transmission over the device's antenna array and connected RF chains
        antenna_transmissions = self.antennas.transmit(signal, self.rf_chain)

        # Simulate mutual coupling behaviour
        coupled_signal = self.coupling.transmit(antenna_transmissions)

        # Return result
        return coupled_signal

    def generate_output(
        self,
        operator_transmissions: List[Transmission] | None = None,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceOutput:
        """Generate the simulated device's output.

        Args:

            operator_transmissions (List[Transmissions], optional):
                List of operator transmissions from which to generate the output.
                If the `operator_transmissions` are not provided, the transmitter's caches will be queried.

            cache (bool, optional):
                Cache the generated output at this device.
                Enabled by default.

            trigger_realization (TriggerRealization, optional):
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

        Returns: The device's output.

        Raises:

            RuntimeError: If no `operator_transmissions` were provided and an operator has no cached transmission.
        """

        # Generate operator transmissions if None were provided:
        _operator_transmissions = (
            [o.transmission for o in self.transmitters]
            if operator_transmissions is None
            else operator_transmissions
        )

        # Query the intended transmission ports of each operator
        operator_streams = [o.selected_transmit_ports for o in self.transmitters]

        if len(_operator_transmissions) != self.transmitters.num_operators:
            raise ValueError(
                f"Unexpcted amount of operator transmissions provided ({len(_operator_transmissions)} instead of {self.transmitters.num_operators})"
            )

        # Generate emerging signals
        emerging_signals: List[Signal] = []

        # If operator separation is enabled, each operator transmission is processed independetly
        if self.operator_separation:
            emerging_signals = [self._simulate_output(t.signal) for t in _operator_transmissions]

        # If operator separation is disable, the transmissions are superimposed to a single signal model
        else:
            superimposed_signal = Signal.Empty(
                self.sampling_rate,
                self.num_transmit_ports,
                carrier_frequency=self.carrier_frequency,
            )

            for transmission, stream_indices in zip(_operator_transmissions, operator_streams):
                if transmission:
                    superimposed_signal.superimpose(
                        transmission.signal, stream_indices=stream_indices
                    )

            emerging_signals = [self._simulate_output(superimposed_signal)]

        # Generate a new trigger realization of none was provided
        if trigger_realization is None:
            trigger_realization = self.trigger_model.realize()

        # Compute padded zeros resulting from the trigger realization delay
        trigger_padding = np.zeros(
            (
                self.antennas.num_transmit_antennas,
                trigger_realization.compute_num_offset_samples(self.sampling_rate),
            ),
            dtype=complex,
        )
        for signal in emerging_signals:
            signal.set_samples(np.append(trigger_padding, signal[:, :], axis=1))

        # Genreate the output data object
        output = SimulatedDeviceOutput(
            emerging_signals,
            trigger_realization,
            self.sampling_rate,
            self.num_transmit_antennas,
            self.carrier_frequency,
        )

        # Cache the output if the respective flag is enabled
        if cache:
            self.__output = output

        # Return result
        return output

    def transmit(
        self, cache: bool = True, trigger_realization: TriggerRealization | None = None
    ) -> SimulatedDeviceTransmission:
        """Transmit over this device.

        Args:

            cache (bool, optional):
                 Cache the transmitted information.
                 Enabled by default.

            trigger_realization (TriggerRealization, optional):
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

        Returns: Information transmitted by this device.
        """

        # Generate operator transmissions
        transmissions = self.transmit_operators()

        # Generate base device output
        output = self.generate_output(transmissions, cache, trigger_realization)

        # Cache and return resulting transmission
        simulated_device_transmission = SimulatedDeviceTransmission.From_SimulatedDeviceOutput(
            output, transmissions
        )

        return simulated_device_transmission

    @property
    def realization(self) -> SimulatedDeviceReceiveRealization | None:
        """Most recent random realization of a receive process.

        Updated during :meth:`.realize_reception`.

        Returns:
            The realization.
            `None` if :meth:`.realize_reception` has not been called yet.
        """

        return self.__realization

    @property
    def output(self) -> SimulatedDeviceOutput | None:
        """Most recent output of this device.

        Updated during :meth:`.transmit`.

        Returns:
            The output information.
            `None` if :meth:`.transmit` has not been called yet.
        """

        return self.__output

    @property
    def input(self) -> ProcessedSimulatedDeviceInput | None:
        """Most recent input of this device.

        Updated during :meth:`.receive` and :meth:`.receive_from_realization`.

        Returns:
            The input information.
            `None` if :meth:`.receive` or :meth:`.receive_from_realization` has not been called yet.
        """

        return self.__input

    def realize_reception(
        self,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        cache=True,
    ) -> SimulatedDeviceReceiveRealization:
        """Generate a random realization for receiving over the simulated device.

        Args:

            noise_level (NoiseLevel, optional):
                Level of the simulated hardware noise.
                If not specified, the device's configured noise level will be assumed.

            noise_model (NoiseModel, optional):
                Model of the simulated hardware noise.
                If not specified, the device's configured noise model will be assumed.

            cache (bool, optional):
                Cache the generated realization at this device.
                Enabled by default.

        Returns: The generated realization.
        """

        # Generate a realization of the noise model
        _noise_level = self.noise_level if noise_level is None else noise_level
        _noise_model = self.noise_model if noise_model is None else noise_model
        noise_realization = _noise_model.realize(_noise_level.get_power())

        # Return device receive realization
        realization = SimulatedDeviceReceiveRealization(noise_realization)

        # Cache realization if the respective flag is enabled
        if cache:
            self.__realization = realization

        return realization

    def _generate_receiver_input(
        self,
        receiver: Receiver,
        baseband_signal: Signal,
        noise_realization: NoiseRealization,
        cache: bool,
    ) -> Signal:
        # ToDo: Handle operator separation

        # Select the appropriate signal streams
        receive_port_selection = (
            slice(None)
            if receiver.selected_receive_ports is None
            else receiver.selected_receive_ports
        )
        received_samples = baseband_signal[receive_port_selection, :]  # type: ignore
        received_signal = baseband_signal.from_ndarray(received_samples)

        # Add noise to the received signal
        noisy_signal = noise_realization.add_to(received_signal)

        # Simulate ADC behaviour
        quantized_signal = self.antennas.analog_digital_conversion(
            noisy_signal, self.rf_chain, receiver.frame_duration
        )

        # Select only the desired signal streams, as specified by the receiver
        receiver_input = Signal.Create(
            quantized_signal[receive_port_selection, :],  # type: ignore
            quantized_signal.sampling_rate,
            quantized_signal.carrier_frequency,
            quantized_signal.delay,
            noise_realization.power,
        )

        # Cache signal and channel state information if the respective flag is enabled
        if cache:
            receiver.cache_reception(receiver_input)

        # The quantized signal is fed into the operator signal processing chain
        return receiver_input

    def process_from_realization(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput,
        realization: SimulatedDeviceReceiveRealization,
        trigger_realization: TriggerRealization | None = None,
        leaking_signal: Signal | None = None,
        cache: bool = True,
    ) -> ProcessedSimulatedDeviceInput:
        """Simulate a signal reception for this device model.

        Args:

            impinging_signals (DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput):
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            realization (SimulatedDeviceRealization):
                Random realization of the device reception process.

            trigger_realization (TriggerRealization, optional):
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

            leaking_signal(Signal, optional):
                Signal leaking from transmit to receive chains.
                If not specified, no leakage is considered during signal reception.

            cache (bool, optional):
                Cache the resulting device reception and operator inputs.
                Enabled by default.

        Returns: The received information.

        Raises:

            ValueError: If `device_signals` is constructed improperly.
        """

        _impinging_signals: Sequence[Signal] = []

        if isinstance(impinging_signals, SimulatedDeviceOutput):
            mixed_signal = impinging_signals.mixed_signal
            _impinging_signals = impinging_signals.emerging_signals

        else:
            if isinstance(impinging_signals, DeviceInput):
                _impinging_signals = impinging_signals.impinging_signals

            elif isinstance(impinging_signals, Signal):
                _impinging_signals = [impinging_signals]

            elif isinstance(impinging_signals, Sequence):
                _impinging_signals = impinging_signals  # type: ignore

            else:
                raise ValueError("Unsupported type of impinging signals")

            mixed_signal = Signal.Empty(
                sampling_rate=self.sampling_rate,
                num_streams=self.num_receive_antennas,
                num_samples=0,
                carrier_frequency=self.carrier_frequency,
            )
            for signal in _impinging_signals:
                mixed_signal.superimpose(signal)

        # Correct the trigger realization
        num_trigger_offset_samples = (
            0
            if trigger_realization is None
            else trigger_realization.compute_num_offset_samples(self.sampling_rate)
        )
        mixed_signal.set_samples(mixed_signal[:, num_trigger_offset_samples:])

        # Model the configured antenna array's input behaviour
        antenna_outputs = self.antennas.receive(
            mixed_signal, self.rf_chain, leaking_signal, self.coupling
        )

        # Generate individual operator inputs
        operator_inputs = [self._generate_receiver_input(r, antenna_outputs, realization.noise_realization, cache) for r in self.receivers]  # type: ignore

        # Generate output information
        processed_input = ProcessedSimulatedDeviceInput(
            _impinging_signals,
            None,
            antenna_outputs,
            self.operator_separation,
            operator_inputs,
            realization.noise_realization,
            trigger_realization,
        )

        # Cache information if respective flag is enabled
        if cache:
            self.__input = processed_input

        # Return final result
        return processed_input

    def process_input(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        leaking_signal: Signal | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        """Process input signals at this device.

        Args:

            impinging_signals (DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput):
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            cache (bool, optional):
                Cache the resulting device reception and operator inputs.
                Enabled by default.

            trigger_realization (TriggerRealization, optional):
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

            noise_level (NoiseLevel, optional):
                Level of the simulated hardware noise.
                If not specified, the device's configured noise level will be assumed.

            noise_model (NoiseModel, optional):
                Model of the simulated hardware noise.
                If not specified, the device's configured noise model will be assumed.

            leaking_signal(Signal, optional):
                Signal leaking from transmit to receive chains.

        Returns: The processed device input.
        """

        # Realize the random process
        realization = self.realize_reception(noise_level, noise_model, cache)

        # Receive the signal
        processed_input = self.process_from_realization(
            impinging_signals, realization, trigger_realization, leaking_signal, cache
        )

        # Return result
        return processed_input

    def receive(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceReception:
        """Receive information at this device.

        Args:

            impinging_signals (DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput):
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            cache (bool, optional):
                Cache the resulting device reception and operator inputs.
                Enabled by default.

            trigger_realization (TriggerRealization, optional):
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

        Returns: The processed device input.
        """

        # Process input
        processed_input = self.process_input(
            impinging_signals, cache=cache, trigger_realization=trigger_realization
        )

        # Genersate receptions
        receptions = self.receive_operators(processed_input, cache)

        # Generate device reception
        return SimulatedDeviceReception.From_ProcessedSimulatedDeviceInput(
            processed_input, receptions
        )
