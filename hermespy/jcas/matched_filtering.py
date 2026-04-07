# -*- coding: utf-8 -*-

from __future__ import annotations
from math import ceil
from typing import Sequence
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light
from scipy.signal import correlate, correlation_lags

from hermespy.beamforming import ReceiveBeamformer
from hermespy.core import (
    AntennaMode,
    DeserializationProcess,
    Receiver,
    ReceiveState,
    Signal,
    Serializable,
    SerializationProcess,
    TransmitState,
    Transmitter,
)
from hermespy.modem import TransmittingModemBase, ReceivingModemBase, CommunicationWaveform
from hermespy.radar import RadarDetector, RadarReception, RadarCube
from .jcas import DuplexJCASOperator, JCASTransmission, JCASReception

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MatchedFilterJcas(
    DuplexJCASOperator[CommunicationWaveform],
    Transmitter[JCASTransmission],
    Receiver[JCASReception],
    Serializable,
):
    """Joint Communication and Sensing Operator.

    A combination of communication and sensing operations.
    Senses the enviroment via a correlatiom-based time of flight estimation of transmitted waveforms.
    """

    _DEFAULT_MAX_VELOCITY = 0.0
    _DEFAULT_VELOCITY_RESOLUTION = 1.0

    # The specific required sampling rate
    __last_transmission: JCASTransmission | None = None
    __max_range: float  # Maximally detectable range

    def __init__(
        self,
        max_range: float,
        waveform: CommunicationWaveform | None = None,
        receive_beamformer: ReceiveBeamformer | None = None,
        detector: RadarDetector | None = None,
        min_range: float = DuplexJCASOperator._DEFAULT_MIN_RANGE,
        max_velocity: float = _DEFAULT_MAX_VELOCITY,
        velocity_resolution: float = _DEFAULT_VELOCITY_RESOLUTION,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        carrier_frequency: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            max_range:
                Maximally detectable range in m.

            waveform:
                Communication waveform used for transmission.
                If not specified, transmitting or receiving will not be possible.

            receive_beamformer:
                Beamforming applied during signal reception.
                If not specified, no beamforming will be applied during reception.

            detector:
                Detector routine configured to generate point clouds from radar cubes.
                If not specified, no point cloud will be generated during reception.

            min_range:
                Minimal range considered for the generated radar cubes.
                Zero by default, but can be adjusted to ignore, for example, self-interference.

            selected_transmit_ports:
                Indices of antenna ports selected for transmission from the operated :class:`Device's<hermespy.core.device.Device>` antenna array.
                If not specified, all available ports will be considered.

            selected_receive_ports:
                Indices of antenna ports selected for reception from the operated :class:`Device's<hermespy.core.device.Device>` antenna array.
                If not specified, all available antenna ports will be considered.

            carrier_frequency:
                Central frequency of the mixed signal in radio-frequency transmission band.
                If not specified, the operated :class:`Device's<hermespy.core.device.Device>` default carrier frequency will be assumed during signal processing.

            seed:
                Random seed used to initialize the pseudo-random number generator.
        """

        # Initialize base class
        DuplexJCASOperator.__init__(
            self,
            waveform,
            receive_beamformer,
            detector,
            min_range,
            selected_transmit_ports,
            selected_receive_ports,
            carrier_frequency,
            seed,
        )

        # Initialize class attributes
        self.__last_transmission = None
        self.max_range = max_range
        self.max_velocity = max_velocity
        self.velocity_resolution = velocity_resolution

    @property
    def power(self) -> float:
        return 0.0 if self.waveform is None else self.waveform.power

    @override
    def notify_transmit_callbacks(self, transmission: JCASTransmission) -> None:
        # Cache the transmission
        # This is required to replay the matched filtering algorithm from datasets
        self.__last_transmission = transmission

        # Propagate to the base notify routine
        super().notify_transmit_callbacks(transmission)

    @override
    def _transmit(self, state: TransmitState, duration: float) -> JCASTransmission:
        return JCASTransmission(TransmittingModemBase._transmit(self, state, duration))

    @override
    def _receive(self, signal: Signal, state: ReceiveState) -> JCASReception:
        # There must be a recent transmission being cached in order to correlate
        if self.__last_transmission is None:
            raise RuntimeError(
                "Receiving from a matched filter joint must be preceeded by a transmission"
            )

        # Receive information
        communication_reception = ReceivingModemBase._receive(self, signal, state)

        resolution = self.range_resolution(state.sampling_rate)
        num_propagated_samples = int(self.max_range / resolution)

        # Append additional samples if the signal is too short
        required_num_received_samples = (
            self.__last_transmission.signal.num_samples + num_propagated_samples
        )
        if signal.num_samples < required_num_received_samples:
            signal = signal.append_samples(
                np.zeros(
                    (signal.num_streams, required_num_received_samples - signal.num_samples),
                    dtype=complex,
                )
            )
        elif signal.num_samples > required_num_received_samples:
            signal = signal[:, :required_num_received_samples]

        # Digital receive beamformer
        angle_bins, beamformed_samples = self._receive_beamform(signal, state)

        # Predict the signal transmitted towards the angles of interest
        if self.__last_transmission.signal.num_streams > 1:
            transmitted_samples = np.empty(
                (angle_bins.shape[0], self.__last_transmission.signal.num_samples),
                dtype=np.complex128,
            )
            for t, angle in enumerate(angle_bins):
                phase_response = state.antennas.horizontal_phase_response(
                    self.__last_transmission.signal.carrier_frequency,
                    angle[0],
                    angle[1],
                    AntennaMode.TX,
                )
                transmitted_samples[t, :] = (
                    phase_response.conj() @ self.__last_transmission.signal.view(np.ndarray)
                )

        # For a single transmit antenna, this is not required
        else:
            transmitted_samples = self.__last_transmission.signal.view(np.ndarray)[[0], :]

        doppler_bins = state.carrier_frequency * np.linspace(-self.max_velocity, self.max_velocity, int(2 * self.max_velocity / self.velocity_resolution) + 1, endpoint=True) / speed_of_light
        cube_data = np.empty((angle_bins.shape[0], doppler_bins.shape[0], num_propagated_samples), dtype=np.float64)

        for d, doppler_shift in enumerate(doppler_bins):
            # Apply Doppler shift to the transmitted signal
            doppler_shifted_samples = transmitted_samples * np.exp(4j * np.pi * doppler_shift * np.arange(transmitted_samples.shape[1]) / state.sampling_rate)

            cube_data[:, d, :] = abs(correlate(beamformed_samples, doppler_shifted_samples, mode="valid", method="fft")[:, :num_propagated_samples])

        # Normalize and convert to power
        cube_data = cube_data**2 / self.__last_transmission.signal.num_samples

        lags = correlation_lags(
            beamformed_samples.shape[1], transmitted_samples.shape[1], mode="valid"
        )

        # Filter range for the minimum range
        min_range_bin = ceil(2 * self.min_range / resolution)

        # Create the cube object
        range_bins = lags[min_range_bin:num_propagated_samples] * resolution
        minmal_cube_data = cube_data[:, :, min_range_bin:num_propagated_samples]
        cube = RadarCube(minmal_cube_data, angle_bins, doppler_bins, range_bins, state.carrier_frequency)

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        radar_reception = RadarReception(signal, cube, cloud)
        jcas_reception = JCASReception(communication_reception, radar_reception)
        return jcas_reception

    def range_resolution(self, sampling_rate: float) -> float:
        """Resolution of the Range Estimation.

        Args:
            sampling_rate:
                Sampling rate of the transmitted and received signal in Hz.
                Defined as the waveform's bandwidth times the oversampling factor.

        Returns:
            float: Resolution in m.

        Raises:
            ValueError:
                If the range resolution is smaller or equal to zero.
        """

        return speed_of_light / (2 * sampling_rate)

    @override
    def frame_duration(self, bandwidth: float) -> float:
        if self.waveform is None:
            return 0.0
        return self.waveform.frame_duration(bandwidth)

    @override
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        if self.waveform is None:
            return 0
        return self.waveform.samples_per_frame(bandwidth, oversampling_factor)

    @property
    def max_range(self) -> float:
        """Maximally estimated reange in m.

        Raises:
            ValueError:
                If `max_range` is smaller or equal to zero.
        """

        return self.__max_range

    @max_range.setter
    def max_range(self, value) -> None:
        if value <= 0.0:
            raise ValueError("Maximum range must be greater than zero")

        self.__max_range = value

    @property
    def max_velocity(self) -> float:
        """Maximal absolute detectable velocity in m/s.

        Raises:
            ValueError: If `max_velocity` is smaller than zero.
        """

        return self.__max_velocity

    @max_velocity.setter
    def max_velocity(self, value) -> None:
        if value < 0.0:
            raise ValueError("Maximum velocity must be greater than or equal to zero")

        self.__max_velocity = value

    @property
    def velocity_resolution(self) -> float:
        """Velocity resolution in m/s.

        Raises:
            ValueError: If `velocity_resolution` is smaller or equal to zero.
        """

        return self.__velocity_resolution

    @velocity_resolution.setter
    def velocity_resolution(self, value) -> None:
        if value <= 0.0:
            raise ValueError("Velocity resolution must be greater than zero")

        self.__velocity_resolution = value

    @override
    def serialize(self, process: SerializationProcess) -> None:
        DuplexJCASOperator.serialize(self, process)
        process.serialize_floating(self.max_range, "max_range")
        process.serialize_floating(self.max_velocity, "max_velocity")
        process.serialize_floating(self.velocity_resolution, "velocity_resolution")
        if self.waveform is not None:
            process.serialize_object(self.waveform, "waveform")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> MatchedFilterJcas:
        return MatchedFilterJcas(
            process.deserialize_floating("max_range"),
            process.deserialize_object("waveform", CommunicationWaveform, None),
            max_velocity=process.deserialize_floating("max_velocity", cls._DEFAULT_MAX_VELOCITY),
            velocity_resolution=process.deserialize_floating("velocity_resolution", cls._DEFAULT_VELOCITY_RESOLUTION),
            **cls._DeserializeParameters(process),  # type: ignore[arg-type]
        )
