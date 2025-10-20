# -*- coding: utf-8 -*-

from __future__ import annotations
from math import ceil
from typing import Sequence, Type
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light
from scipy.fft import ifft, fft, ifftshift

from hermespy.beamforming import ReceiveBeamformer
from hermespy.core import DeserializationProcess, Serializable, ReceiveState, Signal, TransmitState
from hermespy.jcas.jcas import JCASReception, JCASTransmission
from hermespy.modem import OFDMWaveform, ReceivingModemBase, TransmittingModemBase, Symbols
from hermespy.radar import RadarCube, RadarDetector, RadarReception
from .jcas import DuplexJCASOperator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OFDMRadar(DuplexJCASOperator[OFDMWaveform], Serializable):
    """A joint communication and sensing approach estimating a range-power profile from OFDM symbols.

    Refer to :footcite:p:`2009:sturm` for the original publication.
    """

    __last_transmission: JCASTransmission | None = None

    def __init__(
        self,
        waveform: OFDMWaveform | None = None,
        receive_beamformer: ReceiveBeamformer | None = None,
        detector: RadarDetector | None = None,
        min_range: float = DuplexJCASOperator._DEFAULT_MIN_RANGE,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        carrier_frequency: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            waveform:
                Communication waveform emitted by this operator.

            receive_beamformer:
                Beamformer used to process the received signal.

            detector:
                Detector used to process the radar cube.

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
                If not specified, the operated device's default carrier frequency will be assumed during signal processing.

            seed:
                Random seed used to initialize the pseudo-random number generator.
        """

        # Initalize base class
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

    def max_range(self, bandwidth: float) -> float:
        """Maximum range detectable by OFDM radar.

        Defined by equation (12) in :footcite:p:`2009:sturm` as

        .. math::

           d_\\mathrm{max} = \\frac{c_0}{2 \\Delta f} \\quad \\text{.}

        Args:
            bandwidth: Target bandwidth of the OFDM waveform in Hz.

        Returns:
            Maximum range in m.
        """

        if self.waveform is None:
            return 0.0

        return speed_of_light * self.waveform.num_subcarriers / (2 * bandwidth)

    def range_resolution(self, bandwidth: float) -> float:
        """Range resolution achievable by OFDM radar.

        Defined by equation (13) in :footcite:p:`2009:sturm` as

        .. math::

            \\Delta r = \\frac{c_0}{2 B} = \\frac{c_0}{2 N \\Delta f} = \\frac{d_{\\mathrm{max}}}{N} \\quad \\text{.}

        Args:
            bandwidth: Target bandwidth of the OFDM waveform in Hz.

        Returns:
            Range resolution in m.
        """

        return speed_of_light / (2 * bandwidth)

    def max_relative_doppler(self, bandwidth: float) -> float:
        """The maximum relative doppler shift detectable by the OFDM radar in Hz.

        Args:
            bandwidth: Target bandwidth of the OFDM waveform in Hz.

        Returns:
            Maximum relative doppler shift in Hz.
        """

        # The maximum velocity is the wavelength divided by four times the pulse repetition interval
        max_doppler = 1 / (4 * self.frame_duration(bandwidth))
        return max_doppler

    def relative_doppler_resolution(self, bandwidth: float) -> float:
        """The relative doppler resolution achievable by the OFDM radar in Hz.

        Args:
            bandwidth: Target bandwidth of the OFDM waveform in Hz.

        Returns:
            Relative doppler resolution in Hz.
        """

        # The doppler resolution is the inverse of twice the frame duration
        resolution = 1 / (2 * self.frame_duration(bandwidth))
        return resolution

    @property
    def power(self) -> float:
        return 0.0 if self.waveform is None else self.waveform.power

    @override
    def _transmit(self, device: TransmitState, duration: float) -> JCASTransmission:
        communication_transmission = TransmittingModemBase._transmit(self, device, duration)
        self.__last_transmission = JCASTransmission(communication_transmission)
        return self.__last_transmission

    def __estimate_range(
        self,
        transmitted_symbols: Symbols,
        received_signal: np.ndarray,
        bandwidth: float,
        oversampling_factor: int = 1,
    ) -> np.ndarray:
        """Estiamte the range-power profile of the received signal.

        Args:
            transmitted_symbols: The originally transmitted OFDM symbols.
            received_signal: The received OFDM base-band signal samples.
            bandwidth: Target bandwidth of the OFDM waveform in Hz.
            oversampling_factor: Oversampling factor used during the OFDM modulation.

        Returns: The range-power profile of the received signal.
        """

        # Demodulate the signal received from an angle of interest
        received_symbols = self.waveform.demodulate(received_signal, bandwidth, oversampling_factor)

        # Normalize received demodulated symbols equation (8)
        normalized_symbols = np.divide(
            received_symbols.raw,
            transmitted_symbols.raw,
            np.zeros_like(received_symbols.raw),
            where=np.abs(transmitted_symbols.raw) != 0.0,
        )

        # Estimate range-power profile by equation (10)
        power_profile = ifftshift(fft(ifft(normalized_symbols[0, ::], axis=1), axis=0), axes=0)
        return np.abs(power_profile)

    @override
    def _receive(self, signal: Signal, device: ReceiveState) -> JCASReception:
        if self.__last_transmission is None:
            raise RuntimeError("Unable to receive ")

        # Retrieve the transmitted symbols
        transmitted_symbols = self.waveform.place(self.__last_transmission.symbols)

        # Run the normal communication reception processing
        communication_reception = ReceivingModemBase._receive(self, signal, device)

        # Build a radar cube
        angles_of_interest, beamformed_samples = self._receive_beamform(signal, device)
        range_bins = np.arange(self.waveform.num_subcarriers) * self.range_resolution(
            device.bandwidth
        )
        doppler_bins = np.arange(self.waveform.words_per_frame) * self.relative_doppler_resolution(
            device.bandwidth
        ) - self.max_relative_doppler(device.bandwidth)

        # Filter range for the minimum range
        min_range_index = ceil(self.min_range / self.range_resolution(device.bandwidth))
        selected_range_bins = range_bins[min_range_index:]

        cube_data = np.empty(
            (len(angles_of_interest), len(doppler_bins), len(selected_range_bins)), dtype=float
        )

        # Process the single angular line by the waveform generator
        for angle_idx, line in enumerate(beamformed_samples):
            line_estimate = self.__estimate_range(
                transmitted_symbols, line, device.bandwidth, device.oversampling_factor
            )
            cube_data[angle_idx, ::] = line_estimate[min_range_index:]

        # Create radar cube object
        cube = RadarCube(
            cube_data,
            angles_of_interest,
            doppler_bins,
            selected_range_bins,
            device.carrier_frequency,
        )

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        # Generate reception object
        radar_reception = RadarReception(signal, cube, cloud)
        return JCASReception(communication_reception, radar_reception)

    @classmethod
    @override
    def Deserialize(cls: Type[OFDMRadar], process: DeserializationProcess) -> OFDMRadar:
        return cls(**cls._DeserializeParameters(process))  # type: ignore[arg-type]
