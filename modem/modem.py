from typing import List

from numpy import random as rnd
import numpy as np
from typing import Tuple, Generic, TypeVar, Any

from parameters_parser.parameters_modem import ParametersModem
from parameters_parser.parameters_psk_qam import ParametersPskQam
from parameters_parser.parameters_chirp_fsk import ParametersChirpFsk
from parameters_parser.parameters_ofdm import ParametersOfdm
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder
from modem.waveform_generator_psk_qam import WaveformGeneratorPskQam
from modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk
from modem.waveform_generator_ofdm import WaveformGeneratorOfdm
from modem.rf_chain import RfChain
from modem.coding.repetition_encoder import RepetitionEncoder
from modem.coding.ldpc_encoder import LdpcEncoder
from modem.coding.encoder import Encoder
from modem.coding.encoder_manager import EncoderManager
from modem.coding.encoder_factory import EncoderFactory
from source.bits_source import BitsSource
from channel.channel import Channel

P = TypeVar('P', bound=ParametersModem)


class Modem(Generic[P]):
    """Implements a modem.

    The modem consists of an analog RF chain, a waveform generator, and can be used
    either for transmission or reception of a given technology.

    Attributes:
        param (ParametersModem): Modem-specific parameters.
        waveform_generator (WaveformGenerator): waveform generator object, generates baseband samples.
        rf_chain (RfChain): RF chain object, that models RF impairments.
        power_factor(float):
            if this is a transmit modem, signal is scaled to
            the desired power, depending on the current power factor.
        encoder(Encoder):
    """

    def __init__(self, param: P, source: BitsSource,
                 random_number_gen: rnd.RandomState, tx_modem=None) -> None:
        self.param = param
        self.source = source

        self.encoder_factory = EncoderFactory()
        self.encoder_manager = EncoderManager()

        for encoding_type, encoding_params in zip(
                            self.param.encoding_type, self.param.encoding_params):
            encoder: Encoder = self.encoder_factory.get_encoder(
                encoding_params, encoding_type,
                self.param.technology.bits_in_frame)
            self.encoder_manager.add_encoder(encoder)

        self.waveform_generator: Any
        if isinstance(param.technology, ParametersPskQam):
            self.waveform_generator = WaveformGeneratorPskQam(param.technology)
        elif isinstance(param.technology, ParametersChirpFsk):
            self.waveform_generator = WaveformGeneratorChirpFsk(param.technology)
        elif isinstance(param.technology, ParametersOfdm):
            self.waveform_generator = WaveformGeneratorOfdm(
                param.technology, random_number_gen)
        else:
            raise ValueError(
                "invalid technology in constructor of Modem class")
        # if this is a received modem, link to tx modem must be provided
        self._paired_tx_modem = tx_modem
        self.power_factor = 1.  # if this is a transmit modem, signal is scaled to the desired power, depending on the
        # current power factor

        self.rf_chain = RfChain(param.rf_chain, self.waveform_generator.get_power(), random_number_gen,)


    @property
    def paired_tx_modem(self) -> 'Modem':
        return self._paired_tx_modem

    @paired_tx_modem.setter
    def paired_tx_modem(self, other_modem: 'Modem'):
        self._paired_tx_modem = other_modem

    def send(self, drop_duration: float) -> np.ndarray:
        """Returns an array with the complex baseband samples of a waveform generator.

        The signal may be distorted by RF impairments.

        Args:
            drop_duration (float): Length of signal in seconds.

        Returns:
            np.ndarray:
                Complex baseband samples, rows denoting transmitter antennas and
                columns denoting samples.
        """
        # coded_bits = self.encoder.encoder(data_bits)
        number_of_samples = int(
            np.ceil(
                drop_duration *
                self.param.technology.sampling_rate))
        timestamp = 0
        frame_index = 1

        while timestamp < number_of_samples:
            data_bits_per_frame = self.source.get_bits(
                self.encoder_manager.encoders[0].source_bits)

            encoded_bits_per_frame = self.encoder_manager.encode(data_bits_per_frame)
            encoded_bits_per_frame_flattened = np.array([], dtype=int)
            for block in encoded_bits_per_frame:
                encoded_bits_per_frame_flattened = np.append(
                    encoded_bits_per_frame_flattened, block
                )

            frame, timestamp, initial_sample_num = self.waveform_generator.create_frame(
                timestamp, encoded_bits_per_frame_flattened)
            if frame_index == 1:
                tx_signal, samples_delay = self._allocate_drop_size(
                    initial_sample_num, number_of_samples)

            tx_signal, samples_delay = self._add_frame_to_drop(
                initial_sample_num, samples_delay, tx_signal, frame)
            frame_index += 1

        tx_signal = self.rf_chain.send(tx_signal)
        tx_signal = self._adjust_tx_power(tx_signal)
        return tx_signal

    def _add_frame_to_drop(self, initial_sample_num: int,
                           samples_delay: int, tx_signal: np.ndarray,
                           frame: np.ndarray) -> Tuple[np.ndarray, int]:
        initial_sample_idx = samples_delay + initial_sample_num
        end_sample_idx = initial_sample_idx + frame.shape[1]

        if end_sample_idx > tx_signal.shape[1]:
            # last frame may be larger than allocated space, because of
            # filtering
            tx_signal = np.append(
                tx_signal, np.zeros((self.param.number_of_antennas, end_sample_idx - tx_signal.shape[1])), axis=1)

        tx_signal[:, initial_sample_idx:end_sample_idx] += frame
        return tx_signal, samples_delay

    def _allocate_drop_size(self, initial_sample_num: int,
                            number_of_samples: int) -> Tuple[np.ndarray, int]:
        if initial_sample_num < 0:
            # first frame may start before 0 because of filtering
            samples_delay = -initial_sample_num
        else:
            samples_delay = 0

        tx_signal = np.zeros((self.param.number_of_antennas, number_of_samples - initial_sample_num),
                             dtype=complex)
        return tx_signal, samples_delay

    def _adjust_tx_power(self, tx_signal: np.ndarray) -> np.ndarray:
        """Adjusts power of tx_signal by power factor."""
        if self.param.tx_power != 0:
            power = self.waveform_generator.get_power()

            self.power_factor = self.param.tx_power / power
            tx_signal = tx_signal * np.sqrt(self.power_factor)

        return tx_signal

    def receive(self, input_signal: np.ndarray, noise_var: float) -> List[np.array]:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            input_signal (np.ndarray): Received signal.
            noise_var (float): noise variance (for equalization).

        Returns:
            List[np.array]: Detected bits as a list of data blocks for the drop.
        """
        rx_signal = self.rf_chain.receive(input_signal)

        # normalize signal to expected input power
        rx_signal = rx_signal / np.sqrt(self.paired_tx_modem.power_factor)
        noise_var = noise_var / self.paired_tx_modem.power_factor

        all_bits = list()
        timestamp_in_samples = 0

        while rx_signal.size:
            initial_size = rx_signal.shape[1]
            bits_rx, rx_signal = self.waveform_generator.receive_frame(
                rx_signal, timestamp_in_samples, noise_var)

            if rx_signal.size:
                timestamp_in_samples += initial_size - rx_signal.shape[1]

            if not bits_rx[0] is None:
                bits_rx_decoded = self.encoder_manager.decode(bits_rx)
                all_bits.extend(bits_rx_decoded)
        return all_bits

    def get_bit_energy(self) -> float:
        """Returns the average bit energy of the modulated signal.
        """
        R = self.encoder_manager.code_rate
        return self.waveform_generator.get_bit_energy() * self.power_factor / R

    def get_symbol_energy(self) -> float:
        """Returns the average symbol energy of the modulated signal.
        """
        R = self.encoder_manager.code_rate
        return self.waveform_generator.get_symbol_energy() * self.power_factor / R

    def set_channel(self, channel: Channel):
        self.waveform_generator.set_channel(channel)
