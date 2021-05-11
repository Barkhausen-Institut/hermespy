from typing import Dict, Any, List

import numpy as np
from scipy import stats
from scipy.special import comb

from parameters_parser.parameters_scenario import ParametersScenario
from parameters_parser.parameters_psk_qam import ParametersPskQam
from parameters_parser.parameters_chirp_fsk import ParametersChirpFsk
from parameters_parser.parameters_ofdm import (
    ParametersOfdm, GuardInterval
)


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
    """

    def __init__(self, parameters: ParametersScenario) -> None:
        """initializes an object for obtaining theoretical results.

        Based on the parameters in 'parameters'. Depending on the parameters,
        no theoretical results can be generated.
        """
        self.param = parameters

    def get_results(self, snr_type: str,
                    snr_vector_db: np.ndarray) -> List[Dict[str, Any]]:
        """Returns theoretical results for all the Rx modems defined in the scenario.

        depending on their particular modem and channel configurations.

        Args:
            snr_vector_db (np.ndarray): vector of Eb/N0 values.
        """
        snr_lin = 10 ** (snr_vector_db / 10)
        results: List[Dict[str, Any]] = []
        for modem_idx, modem in enumerate(self.param.rx_modem_params):
            results.append(None)
            channel = self.param.channel_model_params[modem_idx][self.param.rx_modem_params[modem_idx].tx_modem]
            bits_per_symbol = np.log2(modem.technology.modulation_order)

            if isinstance(modem.technology,
                          ParametersPskQam) and channel.multipath_model == 'NONE':
                bits_in_frame = modem.technology.number_data_symbols * \
                    np.log2(modem.technology.modulation_order)

                if snr_type == "EB/N0(DB)":
                    ebn0_lin = snr_lin
                elif snr_type == "ES/N0(DB)":
                    ebn0_lin = snr_lin * \
                        np.log2(modem.technology.modulation_order)
                elif snr_type == "CUSTOM":
                    ebn0_lin = snr_vector_db
                else:
                    raise ValueError('invalid SNR type')

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
                    ser = 4 * (np.sqrt(m) - 1) / np.sqrt(m) * \
                        stats.norm.sf(
                            np.sqrt(3 * bits_per_symbol / (m - 1) * ebn0_lin))
                    ber = ser / bits_per_symbol
                else:
                    raise ValueError("modulation order not supported")

                fer = 1 - (1 - ber) ** bits_in_frame
                results[-1] = {'ber': ber, 'fer': fer,
                               'notes': 'AWGN channel'}

            elif (isinstance(modem.technology, ParametersPskQam) and channel.multipath_model == 'STOCHASTIC' and
                  channel.delays.size == 1 and channel.k_factor_rice[0] == 0):

                if snr_type == "EB/N0(DB)":
                    esn0_lin = snr_lin * bits_per_symbol
                elif snr_type == "ES/N0(DB)":
                    esn0_lin = snr_lin
                elif snr_type == "CUSTOM":
                    esn0_lin = snr_vector_db
                else:
                    raise ValueError('invalid SNR type')

                if modem.technology.modulation_order == 2:
                    # BPSK
                    alpha = 1
                    beta = 2.
                elif modem.technology.modulation_order in [4, 8]:
                    # M-PSK
                    alpha = 2
                    beta = 2 * \
                        np.sin(np.pi / modem.technology.modulation_order) ** 2
                elif modem.technology.modulation_order in [16, 64, 256]:
                    # M-QAM
                    alpha = 4 * (np.sqrt(modem.technology.modulation_order) -
                                 1) / np.sqrt(modem.technology.modulation_order)
                    beta = 3 / (modem.technology.modulation_order - 1)
                else:
                    raise ValueError("modulation order not supported")

                ser = alpha / 2 * \
                    (1 - np.sqrt(beta * esn0_lin / 2 / (1 + beta * esn0_lin / 2)))
                ber = ser / np.log2(modem.technology.modulation_order)

                results[-1] = {'ber': ber, 'notes': 'Rayleigh channel'}

            elif isinstance(modem.technology, ParametersChirpFsk) and channel.multipath_model == 'NONE':

                if snr_type == "EB/N0(DB)":
                    ebn0_lin = snr_lin
                elif snr_type == "ES/N0(DB)":
                    ebn0_lin = snr_lin / bits_per_symbol
                elif snr_type == "CUSTOM":
                    ebn0_lin = snr_vector_db
                else:
                    raise ValueError('invalid SNR type')

                ser = np.zeros(snr_vector_db.shape)  # symbol error rate
                bits_per_symbol = np.log2(modem.technology.modulation_order)

                # calculate BER according do Proakis, Salehi, Digital
                # Communications, 5th edition, Section 4.5
                for idx in range(1, modem.technology.modulation_order):
                    ser += ((-1)**(idx + 1) / (idx + 1) * comb(modem.technology.modulation_order - 1, idx, repetition=False)
                            * np.exp(- (idx * bits_per_symbol) / (idx + 1) * ebn0_lin))

                ber = 2**(bits_per_symbol - 1) / (2**bits_per_symbol - 1) * ser
                fer = 1 - (1 - ser) ** modem.technology.number_data_chirps

                results[-1] = {'ber': ber, 'fer': fer,
                               'notes': 'AWGN channel, non-coherent detection, orthogonal modulation'}

            elif isinstance(modem.technology, ParametersOfdm) and channel.multipath_model == 'NONE':

                if snr_type == "EB/N0(DB)":
                    ebn0_lin = snr_lin
                elif snr_type == "ES/N0(DB)":
                    ebn0_lin = snr_lin / bits_per_symbol
                elif snr_type == "CUSTOM":
                    ebn0_lin = snr_vector_db
                else:
                    raise ValueError('invalid SNR type')

                bits_in_frame = 0

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
                results[-1] = {'ber': ber, 'fer': fer,
                               'notes': 'OFDM, AWGN channel'}

        return results
