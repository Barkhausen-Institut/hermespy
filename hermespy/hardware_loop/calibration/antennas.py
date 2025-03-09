# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import (
    AntennaMode,
    DeserializationProcess,
    SerializationProcess,
    Signal,
    SignalBlock,
)
from ..physical_device import AntennaCalibration, PhysicalDevice
from ..scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ScalarAntennaCalibration(AntennaCalibration):
    """Scalar calibration for antenna arrays."""

    __transmit_correction_weights: (
        np.ndarray
    )  # Scalar weights applied to transmit ports of the antenna array.
    __receive_correction_weights: (
        np.ndarray
    )  # Scalar weights applied to receive ports of the antenna array.

    def __init__(
        self,
        transmit_correction_weights: np.ndarray,
        receive_correction_weights: np.ndarray,
        physical_device: PhysicalDevice | None = None,
    ) -> None:
        """Initialize the scalar calibration.

        Args:

            transmit_correction_weights: Scalar weights applied to transmit ports of the antenna array.
            receive_correction_weights: Scalar weights applied to receive ports of the antenna array.
            physical_device: The physical device to which the calibration belongs.
        """

        # Initialize base class
        super().__init__(physical_device)

        # Initialize class attributes
        self.__transmit_correction_weights = transmit_correction_weights
        self.__receive_correction_weights = receive_correction_weights

    @property
    def transmit_correction_weights(self) -> np.ndarray:
        """Scalar weights applied to transmit ports of the antenna array."""

        return self.__transmit_correction_weights

    @property
    def receive_correction_weights(self) -> np.ndarray:
        """Scalar weights applied to receive ports of the antenna array."""

        return self.__receive_correction_weights

    @override
    def correct_transmission(self, transmission: SignalBlock) -> None:
        if transmission.num_streams != self.__transmit_correction_weights.size:
            raise ValueError(
                "The number of streams in the transmission does not match the number of transmit weights."
            )

        np.copyto(transmission, self.__transmit_correction_weights[:, None] * transmission, "safe")

    @override
    def correct_reception(self, reception: SignalBlock) -> None:
        if reception.num_streams != self.__receive_correction_weights.size:
            raise ValueError(
                "The number of streams in the reception does not match the number of receive weights."
            )

        np.copyto(reception, self.__receive_correction_weights[:, None] * reception, "safe")

    @staticmethod
    def Estimate(
        scenario: PhysicalScenario, device: PhysicalDevice, reference_device: PhysicalDevice
    ) -> ScalarAntennaCalibration:
        """Estimate a scalar calibration for the device using a reference device.

        Args:

            scenario: The scenario in which the calibration is performed.
            device: The device to be calibrated.
            reference_device: The reference device used for calibration.

        Returns: The estimated scalar calibration.
        """

        # Make sure both devices are managed by the scenario
        if device not in scenario.devices:
            raise ValueError("The device to be calibrated is not managed by the scenario.")
        if reference_device not in scenario.devices:
            raise ValueError("The reference device is not managed by the scenario.")

        # Compute the expected antenna array weights
        expected_phase_response_tx = (
            device.antennas.cartesian_phase_response(
                device.carrier_frequency, reference_device.global_position, "global", AntennaMode.TX
            )[:, None]
            @ reference_device.antennas.cartesian_phase_response(
                device.carrier_frequency, device.global_position, "global", AntennaMode.RX
            )[None, :]
        )

        expected_phase_response_rx = (
            reference_device.antennas.cartesian_phase_response(
                reference_device.carrier_frequency, device.global_position, "global", AntennaMode.TX
            )[:, None]
            @ device.antennas.cartesian_phase_response(
                reference_device.carrier_frequency,
                reference_device.global_position,
                "global",
                AntennaMode.RX,
            )[None, :]
        )

        # Use are rectangular signal as calibration signal
        # ToDo: Implement a better waveform here
        num_samples = 1000
        calibration_pulse = np.ones(num_samples, dtype=np.complex128)

        # Probe the transmit chain
        tx_probes = np.empty(
            (
                device.num_digital_transmit_ports,
                reference_device.num_digital_receive_ports,
                num_samples,
            ),
            np.complex128,
        )
        transmit_zeros = np.zeros(
            (reference_device.num_digital_receive_ports, num_samples), dtype=np.complex128
        )
        for n in range(device.num_digital_transmit_ports):
            calibration_waveform = np.zeros(
                (device.num_digital_transmit_ports, num_samples), dtype=np.complex128
            )
            calibration_waveform[n, :] = calibration_pulse
            _, reference_reception = scenario.trigger_direct(
                [
                    Signal.Create(
                        calibration_waveform, device.sampling_rate, device.carrier_frequency
                    ),
                    Signal.Create(
                        transmit_zeros,
                        reference_device.sampling_rate,
                        reference_device.carrier_frequency,
                    ),
                ],
                [device, reference_device],
            )
            tx_probes[n, :, :] = reference_reception.getitem(
                (slice(None), slice(num_samples)), unflatten=True
            )

        # Probe the receive chain
        rx_probes = np.empty(
            (
                reference_device.num_digital_transmit_ports,
                device.num_digital_receive_ports,
                num_samples,
            ),
            np.complex128,
        )
        transmit_zeros = np.zeros(
            (device.num_digital_transmit_ports, num_samples), dtype=np.complex128
        )
        for n in range(reference_device.num_digital_transmit_ports):
            calibration_waveform = np.zeros(
                (reference_device.num_digital_transmit_ports, num_samples), dtype=np.complex128
            )
            calibration_waveform[n, :] = calibration_pulse
            device_reception, _ = scenario.trigger_direct(
                [
                    Signal.Create(transmit_zeros, device.sampling_rate, device.carrier_frequency),
                    Signal.Create(
                        calibration_waveform,
                        device.sampling_rate,
                        reference_device.carrier_frequency,
                    ),
                ],
                [device, reference_device],
            )
            rx_probes[n, :, :] = device_reception.getitem(
                (slice(None), slice(num_samples)), unflatten=True
            )

        # Compute the correction weights
        phase_response_tx = np.mean(tx_probes / calibration_pulse, axis=2)
        transmit_correction_weights = np.mean(
            expected_phase_response_tx / (phase_response_tx / np.abs(phase_response_tx)), axis=1
        )
        phase_response_rx = np.mean(rx_probes / calibration_pulse, axis=2)
        receive_correction_weights = np.mean(
            expected_phase_response_rx / (phase_response_rx / np.abs(phase_response_rx)), axis=0
        )

        return ScalarAntennaCalibration(
            transmit_correction_weights, receive_correction_weights, device
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.transmit_correction_weights, "transmit_correction_weights")
        process.serialize_array(self.receive_correction_weights, "receive_correction_weights")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> ScalarAntennaCalibration:
        return ScalarAntennaCalibration(
            process.deserialize_array("transmit_correction_weights", np.complex128),
            process.deserialize_array("receive_correction_weights", np.complex128),
        )
