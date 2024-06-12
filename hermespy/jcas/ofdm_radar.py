# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.constants import speed_of_light
from scipy.fft import ifft, fft, ifftshift

from hermespy.core import Serializable, Signal
from hermespy.jcas.jcas import JCASReception, JCASTransmission
from hermespy.modem import OFDMWaveform, ReceivingModemBase, TransmittingModemBase, Symbols
from hermespy.radar import RadarCube, RadarReception
from .jcas import DuplexJCASOperator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OFDMRadar(DuplexJCASOperator[OFDMWaveform], Serializable):
    """A joint communication and sensing approach estimating a range-power profile from OFDM symbols.

    Refer to :footcite:p:`2009:sturm` for the original publication.
    """

    yaml_tag = "OFDMRadar"

    @property
    def max_range(self) -> float:
        """Maximum range detectable by OFDM radar.

        Defined by equation (12) in :footcite:p:`2009:sturm` as

        .. math::

           d_\\mathrm{max} = \\frac{c_0}{2 \\Delta f} \\quad \\text{.}

        Returns: Maximum range in m.
        """

        if self.waveform is None:
            return 0.0

        return speed_of_light / (2 * self.waveform.subcarrier_spacing)

    @property
    def range_resolution(self) -> float:
        """Range resolution achievable by OFDM radar.

        Defined by equation (13) in :footcite:p:`2009:sturm` as

        .. math::

            \\Delta r = \\frac{c_0}{2 B} = \\frac{c_0}{2 N \\Delta f} = \\frac{d_{\\mathrm{max}}}{N} \\quad \\text{.}
        """

        if self.waveform is None:
            return 0.0

        return self.max_range / self.waveform.num_subcarriers

    @property
    def max_relative_doppler(self) -> float:
        """The maximum relative doppler shift detectable by the OFDM radar in Hz."""

        # The maximum velocity is the wavelength divided by four times the pulse repetition interval
        max_doppler = 1 / (4 * self.frame_duration)
        return max_doppler

    @property
    def relative_doppler_resolution(self) -> float:
        """The relative doppler resolution achievable by the OFDM radar in Hz."""

        # The doppler resolution is the inverse of twice the frame duration
        resolution = 1 / (2 * self.frame_duration)
        return resolution

    @property
    def power(self) -> float:
        return 0.0 if self.waveform is None else self.waveform.power

    def _transmit(self, duration: float = -1) -> JCASTransmission:
        communication_transmission = TransmittingModemBase._transmit(self, duration=duration)
        jcas_transmission = JCASTransmission(communication_transmission)
        return jcas_transmission

    def __estimate_range(self, transmitted_symbols: Symbols, received_signal: Signal) -> np.ndarray:
        """Estiamte the range-power profile of the received signal.

        Args:
            transmitted_symbols (Symbols): The originally transmitted OFDM symbols.
            received_signal (Signal): The received OFDM base-band signal samples.

        Returns:
            np.ndarray: The range-power profile of the received signal.
        """

        # Demodulate the signal received from an angle of interest
        received_symbols = self.waveform.demodulate(received_signal.getitem(0))

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

    def _receive(self, signal: Signal) -> JCASReception:
        # Retrieve previous transmission
        transmission: JCASTransmission | None = self.transmission

        if transmission is None:
            raise RuntimeError("Unable to receive ")

        # Retrieve the transmitted symbols
        transmitted_symbols = self.waveform.place(transmission.symbols)

        # Run the normal communication reception processing
        communication_reception = ReceivingModemBase._receive(self, signal)

        # Build a radar cube
        angles_of_interest, beamformed_samples = self._receive_beamform(signal)
        range_bins = np.arange(self.waveform.num_subcarriers) * self.range_resolution
        doppler_bins = (
            np.arange(self.waveform.words_per_frame) * self.relative_doppler_resolution
            - self.max_relative_doppler
        )

        cube_data = np.empty(
            (len(angles_of_interest), len(doppler_bins), len(range_bins)), dtype=float
        )

        for angle_idx, line in enumerate(beamformed_samples):
            # Process the single angular line by the waveform generator
            line_signal = signal.from_ndarray(line)
            line_estimate = self.__estimate_range(transmitted_symbols, line_signal)

            cube_data[angle_idx, ::] = line_estimate

        # Create radar cube object
        cube = RadarCube(
            cube_data, angles_of_interest, doppler_bins, range_bins, self.carrier_frequency
        )

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        # Generate reception object
        radar_reception = RadarReception(signal, cube, cloud)
        return JCASReception(communication_reception, radar_reception)
