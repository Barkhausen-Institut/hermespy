# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.constants import speed_of_light
from scipy.signal import correlate, correlation_lags

from hermespy.core import Device, Signal, Serializable
from hermespy.modem import TransmittingModemBase, ReceivingModemBase, CommunicationWaveform
from hermespy.radar import RadarReception, RadarCube
from .jcas import DuplexJCASOperator, JCASTransmission, JCASReception

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MatchedFilterJcas(DuplexJCASOperator[CommunicationWaveform], Serializable):
    """Joint Communication and Sensing Operator.

    A combination of communication and sensing operations.
    Senses the enviroment via a correlatiom-based time of flight estimation of transmitted waveforms.
    """

    yaml_tag = "MatchedFilterJcas"
    property_blacklist = {"slot"}

    # The specific required sampling rate
    __sampling_rate: float | None
    __max_range: float  # Maximally detectable range

    def __init__(self, max_range: float, device: Device | None = None, **kwargs) -> None:
        """
        Args:

            max_range (float):
                Maximally detectable range in m.
        """

        # Initialize base class
        DuplexJCASOperator.__init__(self, device, **kwargs)

        # Initialize class attributes
        self.__sampling_rate = None
        self.max_range = max_range
        self.device = device

    def _transmit(self, duration: float = -1.0) -> JCASTransmission:
        communication_transmission = TransmittingModemBase._transmit(self, duration)
        jcas_transmission = JCASTransmission(communication_transmission)
        return jcas_transmission

    def _receive(self, signal: Signal) -> JCASReception:
        # There must be a recent transmission being cached in order to correlate
        if self.transmission is None:
            raise RuntimeError(
                "Receiving from a matched filter joint must be preceeded by a transmission"
            )

        # Receive information
        communication_reception = ReceivingModemBase._receive(self, signal)

        # Re-sample communication waveform
        signal = signal.resample(self.sampling_rate)

        resolution = self.range_resolution
        num_propagated_samples = int(2 * self.max_range / resolution)

        # Append additional samples if the signal is too short
        required_num_received_samples = (
            self.transmission.signal.num_samples + num_propagated_samples
        )
        if signal.num_samples < required_num_received_samples:
            signal.append_samples(
                Signal(
                    np.zeros(
                        (1, required_num_received_samples - signal.num_samples), dtype=complex
                    ),
                    self.sampling_rate,
                    signal.carrier_frequency,
                )
            )

        # Remove possible overhead samples if signal is too long
        # resampled_signal.samples = re
        # sampled_signal.samples[:, :num_samples]

        correlation = (
            abs(
                correlate(
                    signal.samples, self.transmission.signal.samples, mode="valid", method="fft"
                ).flatten()
            )
            / self.transmission.signal.num_samples
        )
        lags = correlation_lags(
            signal.num_samples, self.transmission.signal.num_samples, mode="valid"
        )

        # Append zeros for correct depth estimation
        # num_appended_zeros = max(0, num_samples - resampled_signal.num_samples)
        # correlation = np.append(correlation, np.zeros(num_appended_zeros))

        # Create the cube object
        angle_bins = np.array([[0.0, 0.0]])
        velocity_bins = np.array([0.0])
        range_bins = 0.5 * lags[:num_propagated_samples] * resolution
        cube_data = np.array([[correlation[:num_propagated_samples]]], dtype=float)
        cube = RadarCube(cube_data, angle_bins, velocity_bins, range_bins, self.carrier_frequency)

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        radar_reception = RadarReception(signal, cube, cloud)
        jcas_reception = JCASReception(communication_reception, radar_reception)
        return jcas_reception

    @property
    def sampling_rate(self) -> float:
        modem_sampling_rate = self.waveform.sampling_rate

        if self.__sampling_rate is None:
            return modem_sampling_rate

        return max(modem_sampling_rate, self.__sampling_rate)

    @sampling_rate.setter
    def sampling_rate(self, value: float | None) -> None:
        if value is None:
            self.__sampling_rate = None
            return

        if value <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def range_resolution(self) -> float:
        """Resolution of the Range Estimation.

        Returns:
            float:
                Resolution in m.

        Raises:

            ValueError:
                If the range resolution is smaller or equal to zero.
        """

        return speed_of_light / self.sampling_rate

    @range_resolution.setter
    def range_resolution(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Range resolution must be greater than zero")

        self.sampling_rate = speed_of_light / value

    @property
    def frame_duration(self) -> float:
        return self.waveform.frame_duration

    @property
    def max_range(self) -> float:
        """Maximally Estimated Range.

        Returns:
            The maximum range in m.

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
