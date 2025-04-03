# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from itertools import chain
from typing import List, Set, Type
from typing_extensions import override

import numpy as np

from hermespy.core import (
    AntennaArrayState,
    DeserializationProcess,
    Device,
    DeviceInput,
    DeviceOutput,
    DeviceReception,
    DeviceState,
    DeviceTransmission,
    Hookable,
    ProcessedDeviceInput,
    RandomNode,
    Transformation,
    Transmission,
    TransmitSignalCoding,
    TransmitStreamEncoder,
    Transmitter,
    Reception,
    Scenario,
    Serializable,
    SerializationProcess,
    Signal,
    Receiver,
    ReceiveSignalCoding,
    ReceiveStreamDecoder,
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
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TriggerRealization(Serializable):
    """Realization of a trigger model.

    A trigger realization will contain the offset between a drop start and the waveform frames contained within the drop.
    """

    __num_offset_samples: int
    __sampling_rate: float

    def __init__(self, num_offset_samples: int, sampling_rate: float) -> None:
        """
        Args:

            num_offset_samples:
                Number of discrete samples between drop start and frame start

            sampling_rate:
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

            sampling_rate:
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

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_offset_samples, "num_offset_samples")
        process.serialize_floating(self.sampling_rate, "sampling_rate")

    @classmethod
    def Deserialize(
        cls: Type[TriggerRealization], process: DeserializationProcess
    ) -> TriggerRealization:
        num_offset_samples = process.deserialize_integer("num_offset_samples")
        sampling_rate = process.deserialize_floating("sampling_rate")
        return cls(num_offset_samples, sampling_rate)


class TriggerModel(Serializable, RandomNode):
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

            device: The device to be controlled by this trigger.
        """

        if device.trigger_model is not self:
            device.trigger_model = self

        self.__devices.add(device)

    def remove_device(self, device: SimulatedDevice) -> None:
        """Remove a device from being controlled by this trigger.

        Args:

            device: The device to be removed.
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

            rng:
                Random number generator used to realize this trigger model.
                If not specified, the object's internal generator will be queried.

        Returns: Realization of the trigger model.
        """
        ...  # pragma: no cover


class StaticTrigger(TriggerModel):
    """Model of a trigger that's always perfectly synchronous with the drop start."""

    def realize(self, rng: np.random.Generator | None = None) -> TriggerRealization:
        if self.num_devices < 1:
            sampling_rate = 1.0

        else:
            sampling_rate = list(self.devices)[0].sampling_rate

        # Static triggers will always consider a perfect trigger at the drop start
        return TriggerRealization(0, sampling_rate)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls: Type[StaticTrigger], process: DeserializationProcess) -> StaticTrigger:
        return cls()


class SampleOffsetTrigger(TriggerModel):
    """Model of a trigger that generates a constant offset of samples between drop start and frame start"""

    __num_offset_samples: int

    def __init__(self, num_offset_samples: int) -> None:
        """
        Args:

            num_offset_samples:
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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_offset_samples, "num_offset_samples")

    @override
    @classmethod
    def Deserialize(
        cls: Type[SampleOffsetTrigger], process: DeserializationProcess
    ) -> SampleOffsetTrigger:
        num_offset_samples = process.deserialize_integer("num_offset_samples")
        return cls(num_offset_samples)


class TimeOffsetTrigger(TriggerModel):
    """Model of a trigger that generates a constant time offset between drop start and frame start.

    Note that the offset is rounded to the nearest smaller integer number of samples,
    depending on the sampling rate of the first controlled device's sampling rate.
    """

    __offset: float

    def __init__(self, offset: float) -> None:
        """
        Args:

            offset:
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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.offset, "offset")

    @override
    @classmethod
    def Deserialize(
        cls: Type[TimeOffsetTrigger], process: DeserializationProcess
    ) -> TimeOffsetTrigger:
        offset = process.deserialize_floating("offset")
        return cls(offset)


class RandomTrigger(TriggerModel):
    """Model of a trigger that generates a random offset between drop start and frame start"""

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls: Type[RandomTrigger], process: DeserializationProcess) -> RandomTrigger:
        return cls()


class SimulatedDeviceOutput(DeviceOutput):
    """Information transmitted by a simulated device"""

    __emerging_signals: Sequence[Signal]
    __trigger_realization: TriggerRealization

    def __init__(
        self,
        emerging_signals: Signal | Sequence[Signal],
        leaking_signals: Signal | Sequence[Signal],
        trigger_realization: TriggerRealization,
        sampling_rate: float,
        num_antennas: int,
        carrier_frequency: float,
    ) -> None:
        """
        Args:

            emerging_signals:
                Signal models emerging from the device.

            leaking_signals:
                Signal models leaking from transmit to receive chains.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.

            sampling_rate:
                Device sampling rate in Hz during the transmission.

            num_antennas:
                Number of transmitting device antennas.

            carrier_frequency:
                Device carrier frequency in Hz.

        Raises:

            ValueError: If `sampling_rate` is greater or equal to zero.
            ValueError: If `num_antennas` is smaller than one.
        """

        _emerging_signals = (
            [emerging_signals] if isinstance(emerging_signals, Signal) else emerging_signals
        )
        _leaking_signals = (
            [leaking_signals] if isinstance(leaking_signals, Signal) else leaking_signals
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
        self.__leaking_signals = _leaking_signals

        # Initialize base class
        DeviceOutput.__init__(self, superimposed_signal)

    @classmethod
    def From_DeviceOutput(
        cls: Type[SimulatedDeviceOutput],
        device_output: DeviceOutput,
        emerging_signals: Signal | Sequence[Signal],
        leaking_signals: Signal | Sequence[Signal],
        trigger_realization: TriggerRealization,
    ) -> SimulatedDeviceOutput:
        """Initialize a simulated device output from its base class.

        Args:

            device_output:
                Device output.

            emerging_signals:
                Signal models emerging from the device.

            leaking_signals:
                Signal models leaking from transmit to receive chains.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.

        Returns: The initialized object.
        """

        return cls(
            emerging_signals,
            leaking_signals,
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
        """Signal models emerging from the device."""

        return self.__emerging_signals

    @property
    def leaking_signals(self) -> Sequence[Signal]:
        """Signal models leaking from transmit to receive chains."""

        return self.__leaking_signals

    @property
    def num_leaking_signals(self) -> int:
        """Number of leaking signals.

        Returns: Number of leaking signals.
        """

        return len(self.__leaking_signals)

    def serialize(self, process: SerializationProcess) -> None:
        DeviceOutput.serialize(self, process)
        process.serialize_object(self.trigger_realization, "trigger_realization")
        process.serialize_object_sequence(self.emerging_signals, "emerging_signals")
        process.serialize_object_sequence(self.leaking_signals, "leaking_signals")

    @classmethod
    def Deserialize(
        cls: Type[SimulatedDeviceOutput], process: DeserializationProcess
    ) -> SimulatedDeviceOutput:
        output = DeviceOutput.Deserialize(process)
        trigger_realization = process.deserialize_object("trigger_realization", TriggerRealization)
        emerging_signals = process.deserialize_object_sequence("emerging_signals", Signal)
        leaking_signals = process.deserialize_object_sequence("leaking_signals", Signal)
        return cls.From_DeviceOutput(output, emerging_signals, leaking_signals, trigger_realization)


class SimulatedDeviceTransmission(DeviceTransmission, SimulatedDeviceOutput):
    """Information generated by transmitting over a simulated device."""

    def __init__(
        self,
        operator_transmissions: Sequence[Transmission],
        emerging_signals: Signal | Sequence[Signal],
        leaking_signals: Signal | Sequence[Signal],
        trigger_realization: TriggerRealization,
        sampling_rate: float,
        num_antennas: int,
        carrier_frequency: float,
    ) -> None:
        """
        Args:

            operator_transmissions:
                Information generated by transmitting over transmit operators.

            emerging_signals:
                Signal models emerging from the device.

            leaking_signals:
                Signal models leaking from transmit to receive chains.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.

            sampling_rate:
                Device sampling rate in Hz during the transmission.

            num_antennas:
                Number of transmitting device antennas.

            carrier_frequency:
                Device carrier frequency in Hz.

        Raises:

            ValueError: If `sampling_rate` is greater or equal to zero.
            ValueError: If `num_antennas` is smaller than one.
        """

        # Initialize base classes
        SimulatedDeviceOutput.__init__(
            self,
            emerging_signals,
            leaking_signals,
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
            output.leaking_signals,
            output.trigger_realization,
            output.sampling_rate,
            output.num_antennas,
            output.carrier_frequency,
        )

    def serialize(self, process: SerializationProcess) -> None:
        SimulatedDeviceOutput.serialize(self, process)
        DeviceTransmission.serialize(self, process)

    @classmethod
    def Deserialize(
        cls: Type[SimulatedDeviceTransmission], process: DeserializationProcess
    ) -> SimulatedDeviceTransmission:
        output = SimulatedDeviceOutput.Deserialize(process)
        device_transmission = DeviceTransmission.Deserialize(process)
        return cls.From_SimulatedDeviceOutput(output, device_transmission.operator_transmissions)


class SimulatedDeviceReceiveRealization(Serializable):
    """Realization of a simulated device reception random process."""

    __noise_realization: NoiseRealization

    def __init__(self, noise_realization: NoiseRealization) -> None:
        """
        Args:

            noise_realization:
                Noise realizations for each receive operator.
        """

        self.__noise_realization = noise_realization

    @property
    def noise_realization(self) -> NoiseRealization:
        """Receive operator noise realizations.

        Returns: Sequence of noise realizations corresponding to the number of registerd receive operators.
        """

        return self.__noise_realization

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.noise_realization, "noise_realization")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SimulatedDeviceReceiveRealization:
        return cls(process.deserialize_object("noise_realization", NoiseRealization))


class ProcessedSimulatedDeviceInput(ProcessedDeviceInput, SimulatedDeviceReceiveRealization):
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
        trigger_realization: TriggerRealization,
        noise_realization: NoiseRealization,
    ) -> None:
        """
        Args:

            impinging_signals:
                Numpy vector containing lists of signals impinging onto the device.

            leaking_signal:
                Signal leaking from transmit to receive chains.

            baseband_signal:
                Baseband signal model from which the operator inputs are generated.

            operator_separation:
                Is the operator separation flag enabled?

            operator_inputs:
                Signal models to be processed by the device's receive DSP algorithms.

            trigger_realization:
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
        ProcessedDeviceInput.__init__(self, _impinging_signals, operator_inputs)
        SimulatedDeviceReceiveRealization.__init__(self, noise_realization)

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ProcessedDeviceInput.serialize(self, process)
        SimulatedDeviceReceiveRealization.serialize(self, process)
        process.serialize_object(self.leaking_signal, "leaking_signal")
        process.serialize_object(self.baseband_signal, "baseband_signal")
        process.serialize_integer(int(self.operator_separation), "operator_separation")
        process.serialize_object(self.trigger_realization, "trigger_realization")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> ProcessedSimulatedDeviceInput:
        processed_input = ProcessedDeviceInput.Deserialize(process)
        return cls(
            processed_input.impinging_signals,
            process.deserialize_object("leaking_signal", Signal),
            process.deserialize_object("baseband_signal", Signal),
            bool(process.deserialize_integer("operator_separation")),
            processed_input.operator_inputs,
            process.deserialize_object("trigger_realization", TriggerRealization),
            process.deserialize_object("noise_realization", NoiseRealization),
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
        trigger_realization: TriggerRealization,
        noise_realization: NoiseRealization,
        operator_receptions: Sequence[Reception],
    ) -> None:
        """
        Args:

            impinging_signals:
                Sequences of signal models impinging onto the device from each linked device.

            leaking_signal:
                Signal leaking from transmit to receive chains.

            baseband_signal:
                Baseband signal model from which the operator inputs are generated.

            operator_separation:
                Is the operator separation flag enabled?

            operator_inputs:
                Signal models to be processed by the device's receive DSP algorithms.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.

            noise_realization:
                Realization of the device's noise model.

            operator_receptions:
                Information inferred from receive operators.
        """

        ProcessedSimulatedDeviceInput.__init__(
            self,
            impinging_signals,
            leaking_signal,
            baseband_signal,
            operator_separation,
            operator_inputs,
            trigger_realization,
            noise_realization,
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

            device:
                The simulated device input.

            operator_receptions:
                Information received by device operators.

        Returns: The initialized object.
        """

        return cls(
            device_input.impinging_signals,
            device_input.leaking_signal,
            device_input.baseband_signal,
            device_input.operator_separation,
            device_input.operator_inputs,
            device_input.trigger_realization,
            device_input.noise_realization,
            operator_receptions,
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ProcessedSimulatedDeviceInput.serialize(self, process)
        DeviceReception.serialize(self, process)

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> SimulatedDeviceReception:
        device_input = ProcessedSimulatedDeviceInput.Deserialize(process)
        device_reception = DeviceReception.Deserialize(process)
        return cls.From_ProcessedSimulatedDeviceInput(
            device_input, device_reception.operator_receptions
        )


class SimulatedDeviceState(DeviceState):
    """Data container representing the immutable physical state of a simulated device during simulation drop generation.

    Generated by calling the :meth:`state<hermespy.simulation.simulated_device.SimulatedDevice.state>` property of a :class:`SimulatedDevice<hermespy.simulation.simulated_device.SimulatedDevice>`.
    """

    def __init__(
        self,
        device_id: int,
        trajectory_sample: TrajectorySample,
        carrier_frequency: float,
        sampling_rate: float,
        num_digital_transmit_ports: int,
        num_digital_receive_ports: int,
        antennas: AntennaArrayState,
    ) -> None:
        """
        Args:

            device_id:
                Unique identifier of the device.

            trajectory_sample:
                Position and orientation of the device in time and space.
                Additionally provide the simulation timestamp.

            carrier_frequency:
                Carrier frequency of the device in Hz.

            sampling_rate:
                Sampling rate of the device in Hz.

            num_digital_transmit_ports:
                Number of digital transmit streams feeding into the device's signal encoding layer.
                This is the number of streams expected to be generated by transmit DSP algorithms by default.

            num_digital_receive_ports:
                Number of digital receive streams emerging from the device's signal decoding layer.
                This the number of streams expected to be feeding into receive DSP algorithms by default.

            antennas:
                State of the device's antenna array.
        """

        # Initialize base class
        DeviceState.__init__(
            self,
            device_id,
            trajectory_sample.timestamp,
            trajectory_sample.pose,
            trajectory_sample.velocity,
            carrier_frequency,
            sampling_rate,
            num_digital_transmit_ports,
            num_digital_receive_ports,
            antennas,
        )


class SimulatedDevice(Device[SimulatedDeviceState], Moveable):
    """Representation of an entity capable of emitting and receiving electromagnetic waves.

    A simulation scenario consists of a collection of devices,
    interconnected by a network of channel models.
    """

    _DEFAULT_CARRIER_FREQUENCY: float = 0.0
    _DEFAULT_SAMPLING_RATE: float = 0.0
    _DEFAULT_OPERATOR_SEPARATION: bool = False

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    __isolation: Isolation
    """Model of the device's transmit-receive isolations"""

    __coupling: Coupling
    """Model of the device's antenna array mutual coupling"""

    __trigger_model: TriggerModel
    """Model of the device's triggering behaviour"""

    __noise_level: NoiseLevel
    __noise_model: NoiseModel
    __scenario: Scenario | None  # Scenario this device is attached to
    __sampling_rate: float | None  # Sampling rate at which this device operate
    __carrier_frequency: float  # Center frequency of the mixed signal in rf-band
    __operator_separation: bool  # Operator separation flag
    __simulated_output_callbacks: Hookable[SimulatedDeviceOutput]
    __simulated_input_callbacks: Hookable[ProcessedSimulatedDeviceInput]

    def __init__(
        self,
        scenario: Scenario | None = None,
        antennas: SimulatedAntennaArray | None = None,
        rf_chain: RfChain | None = None,
        isolation: Isolation | None = None,
        coupling: Coupling | None = None,
        trigger_model: TriggerModel | None = None,
        sampling_rate: float = _DEFAULT_SAMPLING_RATE,
        carrier_frequency: float = _DEFAULT_CARRIER_FREQUENCY,
        operator_separation: bool = _DEFAULT_OPERATOR_SEPARATION,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        pose: Transformation | Trajectory | None = None,
        velocity: np.ndarray | None = None,
        transmit_dsp: Transmitter | Sequence[Transmitter] | None = None,
        receive_dsp: Receiver | Sequence[Receiver] | None = None,
        transmit_encoding: (
            TransmitSignalCoding | Sequence[TransmitStreamEncoder] | TransmitStreamEncoder | None
        ) = None,
        receive_decoding: (
            ReceiveSignalCoding | Sequence[ReceiveStreamDecoder] | ReceiveStreamDecoder | None
        ) = None,
        power: float = Device._DEFAULT_POWER,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            scenario:
                Scenario this device is attached to.
                By default, the device is considered floating.

            antennas:
                Antenna array model of the device.
                By default, a single ideal istropic antenna is assumed.

            rf_chain:
                Model of the device's radio frequency amplification chain.
                If not specified, a chain with ideal hardware models will be assumed.

            isolation:
                Model of the device's transmit-receive isolations.
                By default, perfect isolation is assumed.

            coupling:
                Model of the device's antenna array mutual coupling.
                By default, ideal coupling behaviour is assumed.

            trigger_model:
                The assumed trigger model.
                By default, a :class:`StaticTrigger<hermespy.simulation.simulated_device.StaticTrigger>` is assumed.

            sampling_rate:
                Sampling rate at which this device operates.
                By default, the sampling rate of the first operator is assumed.

            carrier_frequency:
                Center frequency of the mixed signal in rf-band in Hz.
                Zero by default.

            noise_level:
                Level of the decvice's additive noise.
                By default, infinite signal-to-noise ratio (SNR) is assumed.

            noise_model:
                Spectral model of the device's additive noise.
                By default, additive white Gaussian noise is assumed.

            operator_separation:
                Operator separation flag.
                Disabled by default.

            pose:
                Position and orientation of the device in time and space.
                If a `Transformation` is provided, the device is assumed to be static.
                If not specified, the device is assumed to be at the origin with zero velocity.

            velocity:
                Initial velocity of the moveable in local coordinates.
                By default, the moveable is assumed to be resting.
                Only considered if `pose` is a `Transformation`, otherwise the velocity is assumed from the `Trajectory`.

            power:
                Power of the device.
                Assumed to be 1.0 by default.

            seed:
                Seed of the device's pseudo-random number generator.
        """

        # Init base classes
        _trajectory = pose if isinstance(pose, Trajectory) else StaticTrajectory(pose, velocity)
        Device.__init__(
            self,
            transmit_dsp,
            receive_dsp,
            transmit_encoding,
            receive_decoding,
            power,
            _trajectory.sample(0).pose,
            seed,
        )
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

        self.noise_level = SNR(float("inf"), self) if noise_level is None else noise_level  # type: ignore[operator]
        self.noise_model = AWGN() if noise_model is None else noise_model  # type: ignore[operator]
        self.operator_separation = operator_separation
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.__simulated_output_callbacks = Hookable()
        self.__simulated_input_callbacks = Hookable()

    @property
    def simulated_output_callbacks(self) -> Hookable[SimulatedDeviceOutput]:
        """Callbacks notified about generated outputs."""

        return self.__simulated_output_callbacks

    @property
    def simulated_input_callbacks(self) -> Hookable[ProcessedSimulatedDeviceInput]:
        """Callbacks notified about processed inputs."""

        return self.__simulated_input_callbacks

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

    # @register(first_impact="receive_devices", title="Device Noise Level")  # type: ignore[misc]
    @property
    def noise(self) -> float:
        """Shorthand property for accessing the noise level of the device.

        Raises:

            ValueError: For negative noise levels.
        """
        return self.noise_level.level  # type: ignore[operator]

    @noise.setter
    def noise(self, value: float) -> None:
        self.noise_level.level = value  # type: ignore[operator]

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

        if self.__sampling_rate > 0.0:
            return self.__sampling_rate

        sampling_rate = 0.0
        for operator in chain(self.transmitters, self.receivers):
            sampling_rate = max(sampling_rate, operator.sampling_rate)  # type: ignore

        return 1.0 if sampling_rate == 0.0 else sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        if value < 0.0:
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

    def state(self, timestamp: float = 0.0) -> SimulatedDeviceState:
        """Immutable physical state of the device during simulation drop generation.

        Args:

            timestamp:
                Global timestamp in seconds.
                By default, zero is assumed.

        Returns: Device state at the given timestamp.
        """

        trajectory_sample = self.trajectory.sample(timestamp)
        return SimulatedDeviceState(
            id(self),
            trajectory_sample,
            self.carrier_frequency,
            self.sampling_rate,
            self.num_digital_transmit_ports,
            self.num_digital_receive_ports,
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

    def _simulate_output(self, signal: Signal) -> tuple[Signal, Signal]:
        """Simulate a device output over the device's hardware model.

        Args:

            signal: Signal feeding into the hardware chain.

        Returns: Signal emerging from the hardware chain.
        """

        # Simulate transmission over the device's antenna array and connected RF chains
        antenna_transmissions, leaking_signal = self.antennas.transmit(
            signal, self.rf_chain, self.isolation
        )

        # Simulate mutual coupling behaviour
        coupled_signal = self.coupling.transmit(antenna_transmissions)

        # Return result
        return coupled_signal, leaking_signal

    def generate_output(
        self,
        operator_transmissions: Sequence[Transmission],
        state: SimulatedDeviceState | None = None,
        resample: bool = True,
        trigger_realization: TriggerRealization | None = None,
        operator_transmit_ports: Sequence[Sequence[int]] | None = None,
    ) -> SimulatedDeviceOutput:
        """Generate the simulated device's output.

        Args:

            operator_transmissions:
                Resulting transmissions of the device's transmit DSP algorithms.

            state:
                State of the device during the transmission.
                If not specified, the simulated device's state at timestamp zero will be assumed.

            resample:
                Resample the output signal to the device's sampling rate.
                Enabled by default.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

            operator_transmit_ports:
                Intended transmission ports of each transmit DSP algorithm.
                If not specified, the operator's default ports will be assumed.

        Returns: The device's output.
        """

        _state = self.state(0.0) if state is None else state

        # Query the intended transmission ports of each operator
        _operator_transmit_ports = (
            [o.selected_transmit_ports for o in self.transmitters]
            if operator_transmit_ports is None
            else operator_transmit_ports
        )

        if len(operator_transmissions) != len(_operator_transmit_ports):
            raise ValueError(
                f"Unexpcted amount of operator transmissions provided ({len(operator_transmissions)} instead of {self.transmitters.num_operators})"
            )

        # Generate emerging signals
        # Represented by a tuple of the RF signal and the leaking signal
        emerging_signals: list[Signal] = []
        leaking_signals: list[Signal] = []

        # If operator separation is enabled, each operator transmission is processed independetly
        if self.operator_separation:
            for t in operator_transmissions:
                rf_signal, leaking_signal = self._simulate_output(t.signal)
                emerging_signals.append(rf_signal)
                leaking_signals.append(leaking_signal)

        # If operator separation is disable, the transmissions are superimposed to a single signal model
        else:
            superimposed_signal = Signal.Empty(
                self.sampling_rate,
                self.num_digital_transmit_ports,
                carrier_frequency=self.carrier_frequency,
            )

            for transmission, stream_indices in zip(
                operator_transmissions, _operator_transmit_ports
            ):
                if transmission:
                    superimposed_signal.superimpose(
                        transmission.signal, resample=resample, stream_indices=stream_indices
                    )

            # Apply the transmit coding
            encoded_signal = (
                self.transmit_coding.encode_streams(
                    superimposed_signal,
                    _state.transmit_state(list(range(_state.antennas.num_transmit_ports))),
                )
                if _state.num_digital_transmit_ports > 0
                else superimposed_signal
            )

            # Simulate the hardware frontend
            rf_signal, leaking_signal = self._simulate_output(encoded_signal)
            emerging_signals.append(rf_signal)
            leaking_signals.append(leaking_signal)

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
            signal.set_samples(np.append(trigger_padding, signal.getitem(), axis=1))

        # Scale the emerging signals by the device's power
        # This assumes that the expected power of each signal is 1.0
        for signal in emerging_signals:
            for block in signal._blocks:
                block *= self.power  # type: ignore[misc]

        # Genreate the output data object
        output = SimulatedDeviceOutput(
            emerging_signals,
            leaking_signals,
            trigger_realization,
            self.sampling_rate,
            self.num_transmit_antennas,
            self.carrier_frequency,
        )

        # Notify the callbacks
        self.output_callbacks.notify(output)
        self.simulated_output_callbacks.notify(output)

        # Return result
        return output

    def transmit(
        self,
        state: SimulatedDeviceState | None = None,
        notify: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceTransmission:
        """Transmit over this device.

        Args:

            state:
                State of the device during the transmission.
                If not specified, the device's current state will be assumed.

            notify:
                Notify the DSP layer's callbacks about the transmission results.
                Enabled by default.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

        Returns: Information transmitted by this device.
        """

        _state = self.state(0.0) if state is None else state

        # Generate operator transmissions
        transmissions = self.transmit_operators(_state, notify)

        # Generate base device output
        output = self.generate_output(transmissions, _state, True, trigger_realization)

        # Cache and return resulting transmission
        simulated_device_transmission = SimulatedDeviceTransmission.From_SimulatedDeviceOutput(
            output, transmissions
        )

        return simulated_device_transmission

    def realize_reception(
        self, noise_level: NoiseLevel | None = None, noise_model: NoiseModel | None = None
    ) -> SimulatedDeviceReceiveRealization:
        """Generate a random realization for receiving over the simulated device.

        Args:

            noise_level:
                Level of the simulated hardware noise.
                If not specified, the device's configured noise level will be assumed.

            noise_model:
                Model of the simulated hardware noise.
                If not specified, the device's configured noise model will be assumed.

        Returns: The generated realization.
        """

        # Generate a realization of the noise model
        _noise_level = self.noise_level if noise_level is None else noise_level  # type: ignore[operator]
        _noise_model = self.noise_model if noise_model is None else noise_model  # type: ignore[operator]
        noise_realization = _noise_model.realize(_noise_level.get_power())

        # Return device receive realization
        realization = SimulatedDeviceReceiveRealization(noise_realization)
        return realization

    def _generate_receiver_input(
        self,
        receiver: Receiver,
        baseband_signal: Signal,
        noise_realization: NoiseRealization,
        state: SimulatedDeviceState,
    ) -> Signal:
        """Generate the input signal of a single receive DSP algorithm.

        Args:

            receiver:
                The receive DSP algorithm.

            baseband_signal:
                Baseband signal model from which the operator inputs are generated.

            noise_realization:
                Realization of the device's noise model.

            state:
                State of the device during the reception.

        Returns: The input signal of the receive DSP algorithm.
        """
        # ToDo: Handle operator separation

        # Select the appropriate signal streams
        receive_port_selection = (
            slice(None)
            if receiver.selected_receive_ports is None
            else receiver.selected_receive_ports
        )  # type: slice | Sequence[int]
        received_signal = baseband_signal.getstreams(receive_port_selection)

        # Add noise to the received signal
        noisy_signal = noise_realization.add_to(received_signal)

        # Simulate ADC behaviour
        quantized_signal = self.antennas.analog_digital_conversion(
            noisy_signal, self.rf_chain, receiver.frame_duration
        )

        # Apply the receive stream decoding
        decoded_signal = (
            self.receive_coding.decode_streams(
                quantized_signal, state.receive_state(list(range(state.antennas.num_receive_ports)))
            )
            if state.num_digital_receive_ports > 0
            else quantized_signal
        )

        # Select only the desired signal streams, as specified by the receiver
        receiver_input = decoded_signal.getstreams(receive_port_selection)
        receiver_input.noise_power = noise_realization.power

        # The quantized signal is fed into the operator signal processing chain
        return receiver_input

    def process_from_realization(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput,
        realization: SimulatedDeviceReceiveRealization,
        trigger_realization: TriggerRealization | None = None,
        leaking_signals: Signal | Sequence[Signal] | None = None,
        state: SimulatedDeviceState | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        """Simulate a signal reception for this device model.

        Args:

            impinging_signals:
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            realization:
                Random realization of the device reception process.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

            leaking_signal:
                Signal leaking from transmit to receive chains.
                Expected to be a sequence of signals if operator_separation is enabled.
                If not specified, no leakage is considered during signal reception.

            state:
                State of the simulated device during the reception.
                If not specified, the device's state at timestamp zero will be assumed.

        Returns: The received information.

        Raises:

            ValueError: If `device_signals` is constructed improperly.
        """

        _state = self.state(0.0) if state is None else state
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
        mixed_signal.set_samples(
            mixed_signal.getitem((slice(None, None), slice(num_trigger_offset_samples, None)))
        )

        # Model the configured antenna array's input behaviour
        # Hack: Only consider the first leaking signal
        leaking_signal = (
            leaking_signals[0] if isinstance(leaking_signals, Sequence) else leaking_signals
        )

        antenna_outputs = self.antennas.receive(
            mixed_signal, self.rf_chain, leaking_signal, self.coupling
        )

        # Generate individual operator inputs
        operator_inputs = [self._generate_receiver_input(r, antenna_outputs, realization.noise_realization, _state) for r in self.receivers]  # type: ignore

        # Generate output information
        processed_input = ProcessedSimulatedDeviceInput(
            _impinging_signals,
            leaking_signal,
            antenna_outputs,
            self.operator_separation,
            operator_inputs,
            trigger_realization,
            realization.noise_realization,
        )

        # Return final result
        return processed_input

    def process_input(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput,
        state: SimulatedDeviceState | None = None,
        trigger_realization: TriggerRealization | None = None,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        leaking_signals: Signal | Sequence[Signal] | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        """Process input signals at this device.

        Args:

            impinging_signals:
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            state:
                State of the device during the reception.
                If not specified, the device's state at timestamp zero will be assumed.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

            noise_level:
                Level of the simulated hardware noise.
                If not specified, the device's configured noise level will be assumed.

            noise_model:
                Model of the simulated hardware noise.
                If not specified, the device's configured noise model will be assumed.

            leaking_signals:
                Signals leaking from transmit to receive chains.
                Expected to be a sequence of signals if operator_separation is enabled.
                If not specified, no leakage is considered during signal reception.

        Returns: The processed device input.
        """

        _state = self.state(0.0) if state is None else state

        # Realize the random process
        realization = self.realize_reception(noise_level, noise_model)

        # Receive the signal
        processed_input = self.process_from_realization(
            impinging_signals, realization, trigger_realization, leaking_signals, _state
        )

        # Notify the callbacks
        self.input_callbacks.notify(processed_input)
        self.simulated_input_callbacks.notify(processed_input)

        # Return result
        return processed_input

    def receive(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput,
        state: SimulatedDeviceState | None = None,
        notify: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceReception:
        """Receive information at this device.

        Args:

            impinging_signals:
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            state:
                State of the device during the reception.
                If not specified, the device's state at timestamp zero will be assumed.

            notify:
                Notify all registered callbacks within the involved DSP algorithms.
                Enabled by default.

            trigger_realization:
                Trigger realization modeling the time delay between a drop start and frame start.
                Perfect triggering is assumed by default.

        Returns: The processed device input.
        """

        _state = self.state(0.0) if state is None else state

        # Process input
        processed_input = self.process_input(impinging_signals, _state, trigger_realization)

        # Genersate receptions
        receptions = self.receive_operators(processed_input, _state, notify)

        # Generate device reception
        return SimulatedDeviceReception.From_ProcessedSimulatedDeviceInput(
            processed_input, receptions
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        Device.serialize(self, process)
        process.serialize_object(self.rf_chain, "rf_chain")
        process.serialize_object(self.isolation, "isolation")
        process.serialize_object(self.coupling, "coupling")
        process.serialize_object(self.trigger_model, "trigger_model")
        # Note: Deactivated due to circular dependency
        # process.serialize_object(self.noise_level, "noise_level")
        process.serialize_object(self.noise_model, "noise_model")  # type: ignore[operator]
        if self.__sampling_rate is not None:
            process.serialize_floating(self.__sampling_rate, "sampling_rate")
        process.serialize_floating(self.__carrier_frequency, "carrier_frequency")
        process.serialize_integer(int(self.__operator_separation), "operator_separation")

    @classmethod
    @override
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        base_parameters = Device._DeserializeParameters(process)
        base_parameters.update(
            {
                "antennas": process.deserialize_object("antennas", SimulatedAntennaArray, None),
                "rf_chain": process.deserialize_object("rf_chain", RfChain, None),
                "isolation": process.deserialize_object("isolation", Isolation, None),
                "coupling": process.deserialize_object("coupling", Coupling, None),
                "trigger_model": process.deserialize_object("trigger_model", TriggerModel, None),
                "noise_level": process.deserialize_object("noise_level", NoiseLevel, None),
                "noise_model": process.deserialize_object("noise_model", NoiseModel, None),
                "sampling_rate": process.deserialize_floating(
                    "sampling_rate", cls._DEFAULT_SAMPLING_RATE
                ),
                "carrier_frequency": process.deserialize_floating(
                    "carrier_frequency", cls._DEFAULT_CARRIER_FREQUENCY
                ),
                "operator_separation": bool(
                    process.deserialize_integer(
                        "operator_separation", cls._DEFAULT_OPERATOR_SEPARATION
                    )
                ),
            }
        )
        return base_parameters

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SimulatedDevice:
        parameters = cls._DeserializeParameters(process)
        return cls(**parameters)  # type: ignore[arg-type]
