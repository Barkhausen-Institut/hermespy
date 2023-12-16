# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import chain
from typing import Dict, List, Optional, Set, Type

import numpy as np
from h5py import Group

from hermespy.core import (
    Device,
    DeviceInput,
    DeviceOutput,
    DeviceReception,
    DeviceTransmission,
    HDFSerializable,
    Moveable,
    ProcessedDeviceInput,
    RandomNode,
    Transformation,
    Transmission,
    Reception,
    Scenario,
    Serializable,
    Signal,
    Receiver,
    SNRType,
)
from hermespy.channel import ChannelPropagation, DirectiveChannelRealization
from .antennas import SimulatedAntennas, SimulatedIdealAntenna, SimulatedUniformArray
from .noise import Noise, NoiseRealization, AWGN
from .rf_chain.rf_chain import RfChain
from .isolation import Isolation, PerfectIsolation
from .coupling import Coupling, PerfectCoupling

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
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
        """Number of discrete samples between drop start and frame start.

        Returns: Number of samples.
        """

        return self.__num_offset_samples

    @property
    def sampling_rate(self) -> float:
        """Sampling rate at which the realization was generated.

        Returns: Sampling rate in Hz.
        """

        return self.__sampling_rate

    @property
    def trigger_delay(self) -> float:
        """Time between drop start and frame start.

        Returns: Trigger delay in seconds.
        """

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
    """Base class for all trigger models.

    Trigger models handle the time synchronization behaviour of devices:
    Within Hermes, the :class:`Drop` models the highest level of physical layer simulation.
    By default, the first sample of a drop is also considered the first sample of the contained
    simulation / sensing frames.
    However, when multiple devices and links interfer with each other, their individual frame structures
    might be completely asynchronous.
    This modeling can be adressed by shared trigger models, all devices sharing a trigger model will
    be frame-synchronous within the simulated drop, however, different trigger models introduce unique time-offsets
    within the simulation drop.
    """

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
    def realize(self) -> TriggerRealization:
        """Realize a triggering of all controlled devices.

        Returns: Realization of the trigger model.
        """
        ...  # pragma: no cover


class StaticTrigger(TriggerModel, Serializable):
    """Model of a trigger that's always perfectly synchronous with the drop start"""

    yaml_tag = "StaticTrigger"

    def realize(self) -> TriggerRealization:
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

    def realize(self) -> TriggerRealization:
        if self.num_devices < 1:
            raise RuntimeError(
                "Realizing a static trigger requires the trigger to control at least one device"
            )

        sampling_rate = list(self.devices)[0].sampling_rate
        return TriggerRealization(self.num_offset_samples, sampling_rate)


class TimeOffsetTrigger(TriggerModel, Serializable):
    """Model of a trigger that generates a constant time offset between drop start and frame start

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

    def realize(self) -> TriggerRealization:
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

    def realize(self) -> TriggerRealization:
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

        trigger_delay = self._rng.integers(0, max_trigger_delay)
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

            emerging_signals (Signal | Sequenece[Signal]):
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
        superimposed_signal = Signal.empty(
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
    """Realization of a simulated device reception random process"""

    __noise_realizations: Sequence[NoiseRealization]

    def __init__(self, noise_realizations: Sequence[NoiseRealization]) -> None:
        """
        Args:

            noise_realization (Sequence[NoiseRealization]):
                Noise realizations for each receive operator.
        """

        self.__noise_realizations = noise_realizations

    @property
    def noise_realizations(self) -> Sequence[NoiseRealization]:
        """Receive operator noise realizations.

        Returns: Sequence of noise realizations corresponding to the number of registerd receive operators.
        """

        return self.__noise_realizations


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
        noise_realizations: Sequence[NoiseRealization],
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

            noise_realization (Sequence[NoiseRealization]):
                Noise realizations for each receive operator.

            trigger_realization (TriggerRealization):
                Trigger realization modeling the time delay between a drop start and frame start.

        Raises:

            ValueError: If `operator_separation` is enabled and `impinging_signals` contains sublists longer than one.
        """

        _impinging_signals: List[Signal] = []
        if len(impinging_signals) > 0:  # pragma: no cover
            if operator_separation and isinstance(impinging_signals[0], Sequence):  # type: ignore
                for signal_sequence in impinging_signals:
                    superposition = signal_sequence[0] if len(signal_sequence) > 0 else Signal.empty(1.0, 1)  # type: ignore
                    for subsignal in signal_sequence[1:]:  # type: ignore
                        superposition.superimpose(subsignal)

                    _impinging_signals.append(superposition)

            elif not operator_separation and isinstance(impinging_signals[0], Signal):
                _impinging_signals = impinging_signals  # type: ignore

            else:
                raise ValueError("Invalid configuration")

        # Initialize base classes
        SimulatedDeviceReceiveRealization.__init__(self, noise_realizations)
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
        noise_realizations: Sequence[NoiseRealization] = []  # ToDo: Serialize noise realizations
        trigger_realization = TriggerRealization.from_HDF(group["trigger_realization"])

        return cls(
            device_input.impinging_signals,
            leaking_signal,
            baseband_signal,
            operator_separation,
            device_input.operator_inputs,
            noise_realizations,
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
        noise_realizations: Sequence[NoiseRealization],
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

            noise_realization (Sequence[NoiseRealization]):
                Noise realizations for each receive operator.

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
            noise_realizations,
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
            device_input.noise_realizations,
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


class SimulatedDevice(Device, Moveable, Serializable):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.

    .. warning::

       When configuring simulated devices within simulation scenarios,
       channel models may ignore spatial properties such as :func:`.position`, :func:`.orientation` or :func:`.velocity`.
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

    __received_channel_realizations: Dict[Device, DirectiveChannelRealization]
    __output: Optional[SimulatedDeviceOutput]  # Most recent device output
    __input: Optional[ProcessedSimulatedDeviceInput]  # Most recent device input
    __noise: Noise  # Model of the hardware noise
    __scenario: Optional[Scenario]  # Scenario this device is attached to
    __sampling_rate: Optional[float]  # Sampling rate at which this device operate
    __carrier_frequency: float  # Center frequency of the mixed signal in rf-band
    __velocity: np.ndarray  # Cartesian device velocity vector
    __operator_separation: bool  # Operator separation flag
    __realization: Optional[
        SimulatedDeviceReceiveRealization
    ]  # Most recent device receive realization

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        antennas: SimulatedAntennas | None = None,
        rf_chain: Optional[RfChain] = None,
        isolation: Optional[Isolation] = None,
        coupling: Optional[Coupling] = None,
        trigger_model: TriggerModel | None = None,
        sampling_rate: Optional[float] = None,
        carrier_frequency: float = 0.0,
        snr: float = float("inf"),
        snr_type: SNRType = SNRType.PN0,
        pose: Transformation | None = None,
        velocity: np.ndarray | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            antennas (SimulatedAntennas, optional):
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

            pose (Transformation, optional):
                Initial pose of the moveable with respect to its reference coordinate frame.
                By default, a unit transformation is assumed.

            velocity (np.ndarray, optional):
                Initial velocity of the moveable in local coordinates.
                By default, the moveable is assumed to be resting.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base classes
        Device.__init__(self, *args, **kwargs)
        Moveable.__init__(self, pose=pose, velocity=velocity)
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

        self.noise = AWGN()
        self.snr = snr
        self.snr_type = snr_type
        self.operator_separation = False
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.velocity = np.zeros(3, dtype=float)
        self.__received_channel_realizations = {}
        self.__input = None
        self.__output = None
        self.__realization = None

    @property
    def antennas(self) -> SimulatedAntennas:
        """Antenna array model of the simulated device."""

        return self.__antennas

    @antennas.setter
    def antennas(self, value: SimulatedAntennas) -> None:
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

    @property
    def noise(self) -> Noise:
        """Model of the hardware noise.

        Returns:
            Noise: Handle to the noise model.
        """

        return self.__noise

    @noise.setter
    def noise(self, value: Noise) -> None:
        """Set the model of the hardware noise."""

        self.__noise = value
        self.__noise.random_mother = self

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
    def sampling_rate(self, value: Optional[float]) -> None:
        """Set the sampling rate at which the device's analog-to-digital converters operate."""

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

    @property
    def velocity(self) -> np.ndarray:
        return self.__velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        value = value.flatten()

        if len(value) != 3:
            raise ValueError("Velocity vector must be three-dimensional")

        self.__velocity = value

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

        operator_transmissions = (
            self.transmit_operators() if operator_transmissions is None else operator_transmissions
        )

        if len(operator_transmissions) != self.transmitters.num_operators:
            raise ValueError(
                f"Unexpcted amount of operator transmissions provided ({len(operator_transmissions)} instead of {self.transmitters.num_operators})"
            )

        # Generate emerging signals
        emerging_signals: List[Signal] = []

        # If operator separation is enabled, each operator transmission is processed independetly
        if self.operator_separation:
            emerging_signals = [self._simulate_output(t.signal) for t in operator_transmissions]

        # If operator separation is disable, the transmissions are superimposed to a single signal model
        else:
            superimposed_signal = Signal.empty(
                self.sampling_rate,
                self.num_transmit_ports,
                carrier_frequency=self.carrier_frequency,
            )

            for transmission in operator_transmissions:
                superimposed_signal.superimpose(transmission.signal)

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
            signal.samples = np.append(trigger_padding, signal.samples, axis=1)

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
    def snr(self) -> float:
        """Signal to noise ratio at the receiver side.

        Returns:

            Linear ratio of signal to noise power.
        """

        return self.__snr

    @snr.setter
    def snr(self, value: float) -> None:
        if value < 0:
            raise ValueError(
                f"The linear signal to noise ratio must be greater than zero (not {value})"
            )

        self.__snr = value

    @property
    def snr_type(self) -> SNRType:
        """Type of signal to noise ratio assumed by the device model."""

        return self.__snr_type

    @snr_type.setter
    def snr_type(self, value: SNRType) -> None:
        self.__snr_type = value

    @property
    def realization(self) -> SimulatedDeviceReceiveRealization | None:
        """Most recent random realization of a receive process.

        Updated during :meth:`.realize_reception`.

        Returns:
            The realization.
            `None` if :meth:`.realize_reception` has not been called yet.
        """

        return self.__realization

    def channel_realization(
        self, transmitter: SimulatedDevice
    ) -> DirectiveChannelRealization | None:
        """Channel realization between this device and a transmitter.

        Updated during :meth:`.realize_reception`.

            transmitter (SimulatedDevice):
                The transmitter.

        Returns:
            The channel realization.
            `None` if no channel realization is available.
        """

        return self.__received_channel_realizations.get(transmitter, None)

    @property
    def output(self) -> Optional[SimulatedDeviceOutput]:
        """Most recent output of this device.

        Updated during :meth:`.transmit`.

        Returns:
            The output information.
            `None` if :meth:`.transmit` has not been called yet.
        """

        return self.__output

    @property
    def input(self) -> Optional[ProcessedSimulatedDeviceInput]:
        """Most recent input of this device.

        Updated during :meth:`.receive` and :meth:`.receive_from_realization`.

        Returns:
            The input information.
            `None` if :meth:`.receive` or :meth:`.receive_from_realization` has not been called yet.
        """

        return self.__input

    def realize_reception(
        self, snr: float = float("inf"), snr_type: SNRType = SNRType.PN0, cache=True
    ) -> SimulatedDeviceReceiveRealization:
        """Generate a random realization for receiving over the simulated device.

        Args:

            snr (float, optional):
                Signal to noise power ratio.
                Infinite by default, meaning no noise will be added to the received signals.

            snr_type (SNRType, optional):
                Type of signal to noise ratio.

            cache (bool, optional):
                Cache the generated realization at this device.
                Enabled by default.

        Returns: The generated realization.
        """

        if snr == float("inf"):
            snr = self.snr

        # Generate noise realizations for each registered receive operator
        noise_realizations = [
            self.noise.realize(r.noise_power(snr, snr_type)) for r in self.receivers
        ]

        # Return device receive realization
        realization = SimulatedDeviceReceiveRealization(noise_realizations)

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
        received_samples = baseband_signal.samples[receive_port_selection, :]  # type: ignore
        received_signal = Signal(
            received_samples, baseband_signal.sampling_rate, baseband_signal.carrier_frequency
        )

        # Add noise to the received signal
        noisy_signal = self.__noise.add(received_signal, noise_realization)

        # Simulate ADC behaviour
        quantized_signal = self.antennas.analog_digital_conversion(
            noisy_signal, self.rf_chain, receiver.frame_duration
        )

        # Select only the desired signal streams, as specified by the receiver
        receiver_input = Signal(
            quantized_signal.samples[receive_port_selection, :],  # type: ignore
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
        impinging_signals: DeviceInput
        | Signal
        | Sequence[Signal]
        | ChannelPropagation
        | Sequence[ChannelPropagation]
        | SimulatedDeviceOutput,
        realization: SimulatedDeviceReceiveRealization,
        trigger_realization: TriggerRealization | None = None,
        leaking_signal: Signal | None = None,
        cache: bool = True,
    ) -> ProcessedSimulatedDeviceInput:
        """Simulate a signal reception for this device model.

        Args:

            impinging_signals (DeviceInput | Signal | Sequence[Signal] | ChannelPropagation | Sequence[ChannelPropagation] | SimulatedDeviceOutput):
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
        mixed_signal = Signal.empty(
            sampling_rate=self.sampling_rate,
            num_streams=self.num_receive_antennas,
            num_samples=0,
            carrier_frequency=self.carrier_frequency,
        )

        if isinstance(impinging_signals, DeviceInput):
            for i in impinging_signals.impinging_signals:
                mixed_signal.superimpose(i)

            _impinging_signals = impinging_signals.impinging_signals

        elif isinstance(impinging_signals, Signal):
            mixed_signal.superimpose(impinging_signals)
            _impinging_signals = [impinging_signals]

        elif isinstance(impinging_signals, Sequence):
            if len(impinging_signals) > 0:
                if isinstance(impinging_signals[0], ChannelPropagation):
                    propagation_impinging: ChannelPropagation
                    for propagation_impinging in impinging_signals:  # type: ignore
                        _impinging_signals.append(propagation_impinging.signal)  # type: ignore
                        mixed_signal.superimpose(propagation_impinging.signal)
                        self.__received_channel_realizations[propagation_impinging.transmitter] = propagation_impinging.realization  # type: ignore

                elif isinstance(impinging_signals[0], Signal):
                    for device_impinging in impinging_signals:
                        mixed_signal.superimpose(device_impinging)  # type: ignore

                    _impinging_signals = impinging_signals  # type: ignore

                else:
                    raise ValueError("Unsupported type of impinging signals")

        elif isinstance(impinging_signals, ChannelPropagation):
            mixed_signal = impinging_signals.signal
            _impinging_signals = [impinging_signals.signal]
            self.__received_channel_realizations[
                impinging_signals.transmitter
            ] = impinging_signals.realization

        elif isinstance(impinging_signals, SimulatedDeviceOutput):
            mixed_signal = impinging_signals.mixed_signal
            _impinging_signals = impinging_signals.emerging_signals

        else:
            raise ValueError("Unsupported type of impinging signals")

        # Correct the trigger realization
        num_trigger_offset_samples = (
            0
            if trigger_realization is None
            else trigger_realization.compute_num_offset_samples(self.sampling_rate)
        )
        mixed_signal.samples = mixed_signal.samples[:, num_trigger_offset_samples:]

        # Model the configured antenna array's input behaviour
        antenna_outputs = self.antennas.receive(
            mixed_signal, self.rf_chain, leaking_signal, self.coupling
        )

        # Generate individual operator inputs
        operator_inputs = [self._generate_receiver_input(r, antenna_outputs, nr, cache) for r, nr in zip(self.receivers, realization.noise_realizations)]  # type: ignore

        # Generate output information
        processed_input = ProcessedSimulatedDeviceInput(
            _impinging_signals,
            None,
            antenna_outputs,
            self.operator_separation,
            operator_inputs,
            realization.noise_realizations,
            trigger_realization,
        )

        # Cache information if respective flag is enabled
        if cache:
            self.__input = processed_input

        # Return final result
        return processed_input

    def process_input(
        self,
        impinging_signals: DeviceInput
        | Signal
        | Sequence[Signal]
        | ChannelPropagation
        | Sequence[ChannelPropagation]
        | SimulatedDeviceOutput,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
        snr: float = float("inf"),
        snr_type: SNRType | None = None,
        leaking_signal: Signal | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        """Process input signals at this device.

        Args:

            impinging_signals (DeviceInput | Signal | Sequence[Signal] | ChannelPropagation | Sequence[ChannelPropagation] | SimulatedDeviceOutput):
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

            snr (float, optional):
                Signal to noise power ratio.
                Infinite by default, meaning no noise will be added to the received signals.

            snr_type (SNRType, optional):
                Type of signal to noise ratio.
                If not specified, the device's default :attr:`snr_type` will be assumed.

            leaking_signal(Signal, optional):
                Signal leaking from transmit to receive chains.

        Returns: The processed device input.
        """

        _snr_type = self.snr_type if snr_type is None else snr_type

        # Realize the random process
        realization = self.realize_reception(snr, _snr_type, cache=cache)

        # Receive the signal
        processed_input = self.process_from_realization(
            impinging_signals, realization, trigger_realization, leaking_signal, cache
        )

        # Return result
        return processed_input

    def receive(
        self,
        impinging_signals: DeviceInput
        | Signal
        | Sequence[Signal]
        | ChannelPropagation
        | Sequence[ChannelPropagation]
        | SimulatedDeviceOutput,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceReception:
        """Receive information at this device.

        Args:

            impinging_signals (DeviceInput | Signal | Sequence[Signal] | ChannelPropagation | Sequence[ChannelPropagation] | SimulatedDeviceOutput):
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
        receptions = self.receive_operators(processed_input.operator_inputs, cache)

        # Generate device reception
        return SimulatedDeviceReception.From_ProcessedSimulatedDeviceInput(
            processed_input, receptions
        )
