# -*- coding: utf-8 -*-
"""HermesPy theoretical scenario performance."""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from scipy import stats
from scipy.special import comb

from hermespy.core import Scenario, Transmitter, Receiver
from hermespy.modem import WaveformGenerator, ChirpFSKWaveform, FilteredSingleCarrierWaveform
from hermespy.channel import Channel, MultipathFadingChannel

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class TheoreticalResults:
    """Generates theoretical results depending on the simulation parameters

    Theoretical results are available just for some scenarios. Currently the following are available:

    - BPSK/QPSK/8PSK and M-QAM
        - in AWGN channel (apart from BSK/QPSK, only an approximation for high SNR)
        - in Rayleigh-faded channels
    - chirp-FSK (supposing orthogonal modulation with non-coherent detection) in AWGN channel
    - OFDM
        - only in AWGN channel

    The results are returned in a map containing the relevant statistics.

    Attributes:

        __waveform_generators (List[type(WaveformGenerator)]):
            Supported types of waveform generators

        __channels (List[type(Channel)]):
            Supported channel types
    """

    __waveform_generators: List[type(WaveformGenerator)]
    __channels: List[type(Channel)]
    __theory_grid: np.ndarray

    def __init__(self) -> None:
        """Theoretical results generator object initialization.

        Based on the parameters in 'parameters'. Depending on the parameters,
        no theoretical results can be generated.
        """

        # Generate theory callback axes
        self.__waveform_generators = [
            ChirpFSKWaveform, FilteredSingleCarrierWaveform]
        self.__channels = [MultipathFadingChannel, Channel]

        # Generate theory callback lookup table
        self.__theory_grid = np.array([[self.__default_theory for _ in self.__channels]
                                      for _ in self.__waveform_generators])

        # Insert theoretically computable configurations
        self.__theory_grid[0, 1] = TheoreticalResults.__theory_chirpfsk_channel
        self.__theory_grid[1,
                           0] = TheoreticalResults.__theory_pskquam_stochastic
        self.__theory_grid[1, 1] = TheoreticalResults.__theory_pskquam_channel

    def theory(self, scenario: Scenario, snrs: np.ndarray) -> np.ndarray:

        theoretical_results = np.empty(
            (scenario.num_transmitters, scenario.num_receivers), dtype=object)

        channels = scenario.channels

        for tx_id, transmitter in enumerate(scenario.transmitters):
            for rx_id, receiver in enumerate(scenario.receivers):

                theoretical_results[tx_id, rx_id] = self.theoretical_results(transmitter, receiver,
                                                                             channels[tx_id, rx_id], snrs)

        return theoretical_results

    def theoretical_results(self,
                            transmitter: Transmitter,
                            receiver: Receiver,
                            channel: Channel,
                            snrs: np.ndarray) -> Optional[Dict[str, Any]]:

        # ToDo: Fix
        return None

        # Currently, only identical waveform generators are theoretically supported
        if isinstance(transmitter.waveform_generator, type(receiver.waveform_generator)):
            return None

        waveform_type = type(transmitter.waveform_generator)
        channel_type = type(channel)

        if waveform_type not in self.__waveform_generators:
            return None

        if channel_type not in self.__channels:
            return None

        # Find indices corresponding to technologies within the lookup grid
        waveform_index = self.__waveform_generators.index(waveform_type)
        channel_index = self.__channels.index(channel_type)

        # Launch callback
        callback = self.__theory_grid[waveform_index, channel_index]
        return callback(transmitter, receiver, channel, snrs)

    @staticmethod
    def __default_theory(transmitter: Transmitter, receiver: Receiver, channel: Channel, snrs: np.ndarray) -> None:
        """Default theory callback.

        Returns:
            None: To indicate no theory is available.
        """

        return None

    @staticmethod
    def __theory_pskquam_channel(transmitter: Transmitter,
                                 receiver: Receiver,
                                 channel: Channel,
                                 snrs: np.ndarray) -> Optional[Dict[str, Any]]:

        bits_in_frame = receiver.num_data_bits_per_frame
        modulation_order = receiver.waveform_generator.modulation_order
        bits_per_symbol = np.log2(modulation_order)

        ber: Optional[np.ndarray] = None

        # BPSK/QPSK
        if modulation_order in [2, 4]:
            ber = stats.norm.sf(np.sqrt(2 * snrs))

        # M-PSK
        elif modulation_order == 8:
            ber = (2 * stats.norm.sf(np.sqrt(2 * bits_per_symbol * snrs) *
                                     np.sin(np.pi / modulation_order)) / bits_per_symbol)

        # M-QAM
        elif modulation_order in [16, 64, 256]:
            ser = 4 * (np.sqrt(modulation_order) - 1) / np.sqrt(modulation_order) * \
                stats.norm.sf(np.sqrt(3 * bits_per_symbol /
                              (modulation_order - 1) * snrs))
            ber = ser / bits_per_symbol

        else:
            return None

        fer = 1 - (1 - ber) ** bits_in_frame
        return {'ber': ber,
                'fer': fer,
                'notes': 'AWGN channel'}

    @staticmethod
    def __theory_pskquam_stochastic(transmitter: Transmitter,
                                    receiver: Receiver,
                                    channel: MultipathFadingChannel,
                                    snrs: np.ndarray) -> Optional[Dict[str, Any]]:

        if channel.delays.size != 1 or channel.rice_factors[0] != 0:
            return None

        modulation_order = receiver.waveform_generator.modulation_order

        # BPSK
        if modulation_order == 2:

            alpha = 1
            beta = 2.

        # M-PSK
        elif modulation_order in [4, 8]:

            alpha = 2
            beta = 2 * np.sin(np.pi / modulation_order) ** 2

        # M-QAM
        elif modulation_order in [16, 64, 256]:

            alpha = 4 * (np.sqrt(modulation_order) - 1) / \
                np.sqrt(modulation_order)
            beta = 3 / (modulation_order - 1)

        else:
            return None

        ser = alpha / 2 * \
            (1 - np.sqrt(beta * snrs / 2 / (1 + beta * snrs / 2)))
        ber = ser / np.log2(modulation_order)

        return {'ser': ser, 'ber': ber, 'notes': 'Rayleigh channel'}

    @staticmethod
    def __theory_chirpfsk_channel(transmitter: Transmitter,
                                  receiver: Receiver,
                                  channel: Channel,
                                  ebn0_linear: np.ndarray) -> Optional[Dict[str, Any]]:

        mod_order = receiver.waveform_generator.modulation_order

        # For modulation orders greater than 64 the implemented method produces numerical errors
        if mod_order > 64:
            return None

        n_bits = np.log2(mod_order)

        # calculate BER according do Proakis, Salehi, Digital
        # Communications, 5th edition, Section 4.5, Equations 44 and 47
        ser = np.zeros(len(ebn0_linear))  # symbol error rate
        for n in range(2, mod_order+1):
            ser += (-1)**n / n * exp(- (n - 1) * n_bits / n *
                                     ebn0_linear) * comb(mod_order - 1, n - 1)

        # Bit error rate
        ber = 2 ** (n_bits - 1) / (2 ** n_bits - 1) * ser
        fer = 1 - (1 - ser) ** receiver.waveform_generator.num_data_chirps

        return {'ber': ber, 'fer': fer,
                'notes': 'AWGN channel, non-coherent detection, orthogonal modulation'}

    @staticmethod
    def plot_theory_chirpfsk(modulation_orders: np.ndarray, ebn0: np.ndarray) -> None:
        """Visualize the chirp fsk theory via PyPlot.

        Args:
            modulation_orders (np.ndarray): Considered order of modulations.
            ebn0 (np.ndarray): Bit energy to noise power ratios (in dB).
        """

        fig, axes = plt.subplots()
        fig.suptitle(
            "Error Probability Orthogonal Signaling, Noncoherent Detection")

        ebn0_linear = 10 ** (ebn0 / 10)

        for mod_order in modulation_orders:

            n_bits = np.log2(mod_order)  # Number of bits per symbol

            ser = np.zeros(len(ebn0_linear))  # symbol error rate
            for n in range(2, mod_order+1):
                ser += (-1)**n / n * exp(- (n - 1) * n_bits / n *
                                         ebn0_linear) * comb(mod_order - 1, n - 1)

            # Bit error rate
            ber = 2 ** (n_bits - 1) / (2 ** n_bits - 1) * ser

            axes.plot(ebn0, ber, label="M = {}".format(mod_order))

        axes.set_yscale('log')
        axes.set(xlabel='SNR [dB]')
        axes.set(ylabel='Probability of Bit Error')
        axes.grid()
        axes.legend()

    @staticmethod
    def __theory_ofdm_channel(transmitter: Transmitter,
                              receiver: Receiver,
                              channel: Channel,
                              snrs: np.ndarray) -> Optional[Dict[str, Any]]:

        # TODO: Re-implement for new OFDM model
        return None

        """bits_in_frame = 0

        samples_in_frame_no_oversampling = 0
        number_cyclic_prefix_samples = 0
        number_of_data_samples = 0

        for frame_element in modem.technology.frame_structure:
            if isinstance(frame_element, GuardInterval):
                samples_in_frame_no_oversampling += frame_element.no_samples
            else:
                samples_in_frame_no_oversampling += frame_element.cyclic_prefix_samples
                number_cyclic_prefix_samples += frame_element.cyclic_prefix_samples

                samples_in_frame_no_oversampling += frame_element.no_samples
                number_of_data_samples += frame_element.no_samples

        cyclic_prefix_overhead = ((number_of_data_samples + number_cyclic_prefix_samples)
                                  / number_of_data_samples)

        ebn0_lin = ebn0_lin / cyclic_prefix_overhead

        if modem.technology.modulation_order in [2, 4]:
            # BPSK/QPSK
            ber = stats.norm.sf(np.sqrt(2 * ebn0_lin))
        elif modem.technology.modulation_order == 8:
            # M-PSK
            ber = (2 * stats.norm.sf(np.sqrt(2 * bits_per_symbol * ebn0_lin)
                                     * np.sin(np.pi / modem.technology.modulation_order)) / bits_per_symbol)
        elif modem.technology.modulation_order in [16, 64, 256]:
            # M-QAM
            m = modem.technology.modulation_order
            bits_per_symbol = np.log2(m)
            ser = 4 * (np.sqrt(m) - 1) / np.sqrt(m) * stats.norm.sf(
                np.sqrt(3 * bits_per_symbol / (m - 1) * ebn0_lin))
            ber = ser / bits_per_symbol
        else:
            raise ValueError("modulation order not supported")

        fer = 1 - (1 - ber) ** bits_in_frame
        return = {'ber': ber, 'fer': fer,
                       'notes': 'OFDM, AWGN channel'}"""
