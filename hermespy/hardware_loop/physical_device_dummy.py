# -*- coding: utf-8 -*-

from __future__ import annotations
from collections.abc import Sequence
from typing import Type
from typing_extensions import override

import numpy as np

from hermespy.core import (
    DeserializationProcess,
    DeviceInput,
    DeviceState,
    Receiver,
    ReceiveSignalCoding,
    ReceiveStreamDecoder,
    Serializable,
    Signal,
    Transformation,
    Transmission,
    TransmitSignalCoding,
    TransmitStreamEncoder,
    Transmitter,
)
from hermespy.simulation import (
    Coupling,
    NoiseLevel,
    NoiseModel,
    ProcessedSimulatedDeviceInput,
    RfChain,
    Isolation,
    SimulatedAntennaArray,
    SimulatedDevice,
    SimulatedDeviceOutput,
    SimulatedDeviceReception,
    SimulatedDeviceState,
    SimulatedDeviceTransmission,
    SimulationScenario,
    Trajectory,
    TriggerModel,
    TriggerRealization,
)
from .physical_device import (
    AntennaCalibration,
    DelayCalibrationBase,
    LeakageCalibrationBase,
    PhysicalDevice,
    PhysicalDeviceState,
)
from .scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceDummyState(
    SimulatedDeviceState, PhysicalDeviceState
): ...  # pragma: no cover  # noqa: E701


class PhysicalDeviceDummy(SimulatedDevice, PhysicalDevice, Serializable):
    """Physical device dummy for testing and demonstration.

    The physical device dummy always receives back its most recent transmission.
    """

    __receive_transmission: bool
    __uploaded_signal: Signal
    __downloaded_signal: Signal

    def __init__(
        self,
        receive_transmission: bool = True,
        scenario: PhysicalScenarioDummy | None = None,
        antennas: SimulatedAntennaArray | None = None,
        rf_chain: RfChain | None = None,
        isolation: Isolation | None = None,
        coupling: Coupling | None = None,
        trigger_model: TriggerModel | None = None,
        sampling_rate: float = SimulatedDevice._DEFAULT_SAMPLING_RATE,
        carrier_frequency: float = SimulatedDevice._DEFAULT_CARRIER_FREQUENCY,
        operator_separation: bool = SimulatedDevice._DEFAULT_OPERATOR_SEPARATION,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        pose: Transformation | Trajectory | None = None,
        velocity: np.ndarray | None = None,
        max_receive_delay: float = PhysicalDevice._DEFAULT_MAX_RECEIVE_DELAY,
        noise_power: np.ndarray | None = None,
        leakage_calibration: LeakageCalibrationBase | None = None,
        delay_calibration: DelayCalibrationBase | None = None,
        antenna_calibration: AntennaCalibration | None = None,
        adaptive_sampling: bool = False,
        lowpass_filter: bool = False,
        lowpass_bandwidth: float = PhysicalDevice._DEFAULT_LOWPASS_BANDWIDTH,
        transmit_dsp: Transmitter | Sequence[Transmitter] | None = None,
        receive_dsp: Receiver | Sequence[Receiver] | None = None,
        transmit_encoding: TransmitSignalCoding | Sequence[TransmitStreamEncoder] | None = None,
        receive_decoding: ReceiveSignalCoding | Sequence[ReceiveStreamDecoder] | None = None,
        power: float = SimulatedDevice._DEFAULT_POWER,
        seed: int | None = None,
    ) -> None:
        # Initialize base classes
        SimulatedDevice.__init__(
            self,
            scenario,
            antennas,
            rf_chain,
            isolation,
            coupling,
            trigger_model,
            sampling_rate,
            carrier_frequency,
            operator_separation,
            noise_level,
            noise_model,
            pose,
            velocity,
            transmit_dsp,
            receive_dsp,
            transmit_encoding,
            receive_decoding,
            power,
            seed,
        )
        PhysicalDevice.__init__(
            self,
            max_receive_delay,
            noise_power,
            leakage_calibration,
            delay_calibration,
            antenna_calibration,
            adaptive_sampling,
            lowpass_filter,
            lowpass_bandwidth,
            transmit_dsp,
            receive_dsp,
            transmit_encoding,
            receive_decoding,
            power,
            pose.sample(0).pose if isinstance(pose, Trajectory) else pose,
            seed,
        )

        # Initialize internal state
        self.receive_transmission = receive_transmission
        self.__uploaded_signal = Signal.Empty(1.0, self.num_antennas)
        self.__downloaded_signal = Signal.Empty(1.0, self.num_antennas)

    def state(self, timestamp: float = 0.0) -> PhysicalDeviceDummyState:
        trajectory_sample = self.trajectory.sample(timestamp)
        return PhysicalDeviceDummyState(
            id(self),
            trajectory_sample,
            self.carrier_frequency,
            self.sampling_rate,
            self.num_digital_transmit_ports,
            self.num_digital_receive_ports,
            self.antennas.state(trajectory_sample.pose),
        )

    @property
    def receive_transmission(self) -> bool:
        """Whether the device receives back its own transmission."""

        return self.__receive_transmission

    @receive_transmission.setter
    def receive_transmission(self, value: bool) -> None:
        self.__receive_transmission = value

    def _upload(self, signal: Signal) -> None:
        self.__uploaded_signal = signal

    def _download(self) -> Signal:
        return self.__downloaded_signal

    def transmit(
        self,
        state: PhysicalDeviceDummyState | SimulatedDeviceState | None = None,
        notify: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceTransmission:
        # Generate device transmission
        device_transmission = SimulatedDevice.transmit(self, state, notify, trigger_realization)

        # Upload mixed signal
        self._upload(device_transmission.mixed_signal)

        return device_transmission

    def process_input(
        self,
        impinging_signals: (
            DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput | None
        ) = None,
        state: PhysicalDeviceDummyState | SimulatedDeviceState | None = None,
        trigger_realization: TriggerRealization | None = None,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        leaking_signals: Signal | Sequence[Signal] | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        _impinging_signals = (
            self.__uploaded_signal if impinging_signals is None else impinging_signals
        )
        return SimulatedDevice.process_input(
            self,
            _impinging_signals,
            state,
            trigger_realization,
            noise_level,
            noise_model,
            leaking_signals,
        )

    def receive(
        self,
        impinging_signals: (
            DeviceInput | Signal | Sequence[Signal] | SimulatedDeviceOutput | None
        ) = None,
        state: PhysicalDeviceDummyState | SimulatedDeviceState | None = None,
        notify: bool = True,
        trigger_realization: TriggerRealization | None = None,
    ) -> SimulatedDeviceReception:
        if impinging_signals is None:
            impinging_signals = self._download()

        return SimulatedDevice.receive(self, impinging_signals, state, notify, trigger_realization)

    def trigger(self) -> None:
        if self.receive_transmission:
            self.__downloaded_signal = self.__uploaded_signal

        else:
            samples = np.zeros(self.__uploaded_signal.shape)
            self.__downloaded_signal = Signal.Create(
                samples, self.sampling_rate, self.carrier_frequency
            )

    def trigger_direct(self, signal: Signal, calibrate: bool = True) -> Signal:
        if self.receive_transmission:
            input = signal

        else:
            input = Signal.Create(
                np.zeros(
                    (self.antennas.num_receive_antennas, signal.num_samples), dtype=np.complex128
                ),
                self.sampling_rate,
                self.carrier_frequency,
            )

        # Apply the simulation receive model
        leaking_signal = self.isolation.leak(signal)
        processed_input = self.process_input(input, leaking_signals=leaking_signal)
        baseband_signal = processed_input.baseband_signal

        # Apply correction routines if calibrations are available
        corrected_signal = (
            baseband_signal
            if not calibrate or self.leakage_calibration is None
            else self.leakage_calibration.remove_leakage(signal, baseband_signal)
        )

        return corrected_signal

    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate

    @classmethod
    @override
    def Deserialize(
        cls: Type[PhysicalDeviceDummy], process: DeserializationProcess
    ) -> PhysicalDeviceDummy:
        return cls(**cls._DeserializeParameters(process))  # type: ignore[arg-type]


class PhysicalScenarioDummy(
    SimulationScenario, PhysicalScenario[PhysicalDeviceDummy], Serializable
):
    """Physical scenario for testing and demonstration."""

    def __init__(
        self, seed: int | None = None, devices: Sequence[PhysicalDeviceDummy] | None = None
    ) -> None:
        # Initialize base classes
        SimulationScenario.__init__(self, seed=seed, devices=devices)
        PhysicalScenario.__init__(self, seed=seed, devices=devices)

    @classmethod
    @override
    def _device_type(cls) -> type[PhysicalDeviceDummy]:
        return PhysicalDeviceDummy

    def add_device(self, device: SimulatedDevice | PhysicalDeviceDummy) -> None:
        # Adding a device resolves to the simulation scenario's add device method
        SimulationScenario.add_device(self, device)

    def receive_devices(
        self,
        impinging_signals: (
            Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]] | None
        ) = None,
        states: Sequence[DeviceState] | Sequence[SimulatedDeviceState] | None = None,
        notify: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
        leaking_signals: Sequence[Signal] | Sequence[Sequence[Signal]] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        """Process receive layers of all registered devices.

        Resolves to :meth:`PhysicalScenario.receive_devices<hermespy.hardware_loop.scenario.PhysicalScenario.receive_devices>`
        if `impinging_signals` is not provided.
        Otherwise, resolves to :meth:`SimulationScenario.receive_devices<hermespy.simulation.simulation.SimulationScenario.receive_devices>`.
        """

        if impinging_signals is None:
            physical_device_receptions = PhysicalScenario.receive_devices(
                self, impinging_signals, states, False
            )
            impinging_signals = [r.impinging_signals for r in physical_device_receptions]

        return SimulationScenario.receive_devices(
            self, impinging_signals, states, notify, trigger_realizations, leaking_signals  # type: ignore
        )

    def _trigger(self) -> None:
        # Triggering is equivalent to generating a new simulation drop
        SimulationScenario.drop(self)  # type: ignore

    def _trigger_direct(
        self,
        transmissions: list[Signal],
        devices: list[PhysicalDeviceDummy],
        calibrate: bool = True,
        timestamp: float = 0.0,
    ) -> list[Signal]:
        # Realize triggers
        triggers = self.realize_triggers(devices)

        # Generate transmissions considering the hardware models
        device_outputs = [
            device.generate_output(
                [Transmission(transmission)],
                None,
                True,
                trigger,
                [np.arange(device.num_digital_transmit_ports).tolist()],
            )
            for device, transmission, trigger in zip(devices, transmissions, triggers)
        ]

        # Realize all channels
        channel_realizations = self.realize_channels()

        # Propgate over all channels
        propagated_signals = np.empty((len(devices), len(devices)), dtype=object)
        for n, (alpha_device, alpha_transmission) in enumerate(zip(devices, device_outputs)):
            for m, (beta_device, beta_transmission) in enumerate(
                zip(devices[n:], device_outputs[n:]), n
            ):
                # Select the correct channel and its respective realization
                channel_index = self.channels.index(self.channel(alpha_device, beta_device))
                channel_realization = channel_realizations[channel_index]

                # Sample the propagations, optimizing for reciprocal channels
                alpha_sample = channel_realization.sample(
                    alpha_device,
                    beta_device,
                    timestamp,
                    alpha_transmission.carrier_frequency,
                    alpha_transmission.sampling_rate,
                )
                beta_sample = channel_realization.reciprocal_sample(
                    alpha_sample,
                    beta_device,
                    alpha_device,
                    timestamp,
                    beta_transmission.carrier_frequency,
                    beta_transmission.sampling_rate,
                )

                # Propagate
                beta_reception = alpha_sample.propagate(alpha_transmission)
                alpha_reception = beta_sample.propagate(beta_transmission)

                # Store the impinging signals
                propagated_signals[m, n] = beta_reception
                propagated_signals[n, m] = alpha_reception

        # Receive devices
        received_base_band_signals = [
            device.process_input(impinging_signals.tolist(), None, trigger).baseband_signal
            for impinging_signals, device, trigger in zip(propagated_signals, devices, triggers)
        ]

        return received_base_band_signals
