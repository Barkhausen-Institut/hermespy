from __future__ import annotations
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import Type, List
import numpy as np
import numpy.random as rnd

from source import BitsSource
from modem import Modem
from modem.waveform_generator import WaveformGenerator


class Receiver(Modem):

    yaml_tag = 'Receiver'

    def __init__(self, **kwargs) -> None:
        Modem.__init__(self, **kwargs)

    def receive(self, input_signal: np.ndarray, noise_var: float) -> np.ndarray:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            input_signal (np.ndarray): Received signal.
            noise_var (float): noise variance (for equalization).

        Returns:
            np.array: Detected bits as a list of data blocks for the drop.
        """
        rx_signal = self.rf_chain.receive(input_signal)

        # If no receiving waveform generator is configured, no signal is being received
        if self.waveform_generator is None:
            return np.empty(0, dtype=complex)

        # normalize signal to expected input power
        rx_signal = rx_signal / np.sqrt(1.0)  # TODO: Re-implement pair power factor
        noise_var = noise_var / 1.0  # TODO: Re-implement pair power factor

        received_bits = np.empty(0, dtype=int)
        timestamp_in_samples = 0

        while rx_signal.size:
            initial_size = rx_signal.shape[1]
            frame_bits, rx_signal = self.waveform_generator.receive_frame(
                rx_signal, timestamp_in_samples, noise_var)

            if rx_signal.size:
                timestamp_in_samples += initial_size - rx_signal.shape[1]

            received_bits = np.append(received_bits, frame_bits)

        decoded_bits = self.encoder_manager.decode(received_bits)
        return decoded_bits

    @classmethod
    def from_yaml(cls: Type[Receiver], constructor: RoundTripConstructor, node: Node) -> Receiver:

        state = constructor.construct_mapping(node, CommentedOrderedMap)

        waveform_generator = None
        bits_source = None

        for key in state.keys():
            if key.startswith(WaveformGenerator.yaml_tag):
                waveform_generator = state.pop(key)
                break

        for key in state.keys():
            if key.startswith(BitsSource.yaml_tag):
                bits_source = state.pop(key)
                break

        state[WaveformGenerator.yaml_tag] = waveform_generator
        state[BitsSource.yaml_tag] = bits_source

        args = dict((k.lower(), v) for k, v in state.items())

        position = args.pop('position', None)
        if position is not None:
            args['position'] = np.array(position)

        orientation = args.pop('orientation', None)
        if position is not None:
            args['orientation'] = np.array(orientation)

        # Convert the random seed to a new random generator object if its specified within the config
        random_seed = args.pop('random_seed', None)
        if random_seed is not None:
            args['random_generator'] = rnd.default_rng(random_seed)

        return Receiver(**args)

    @property
    def index(self) -> int:
        """The index of this receiver in the scenario.

        Returns:
            int:
                The index.
        """

        return self.scenario.receivers.index(self)

    @property
    def paired_modems(self) -> List[Modem]:
        """The modems connected to this modem over an active channel.

        Returns:
            List[Modem]:
                A list of paired modems.
        """

        return [channel.receiver for channel in self.scenario.arriving_channels(self, True)]
