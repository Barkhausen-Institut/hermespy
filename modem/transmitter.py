# -*- coding: utf-8 -*-
"""HermesPy transmitting modem."""

from __future__ import annotations
from ruamel.yaml import SafeConstructor, Node, MappingNode, ScalarNode
from typing import TYPE_CHECKING, Type, List, Any, Optional
from math import ceil
from scipy.constants import speed_of_light
import numpy as np
import numpy.random as rnd

from modem import Modem
from source import BitsSource
from modem.waveform_generator import WaveformGenerator
from modem.precoding import SymbolPrecoding

if TYPE_CHECKING:
    from channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Transmitter(Modem):

    yaml_tag = 'Transmitter'

    def __init__(self, **kwargs: Any) -> None:
        """Object initialization.

        Args:
            **kwargs (Any): Transmitter configuration.
        """

        Modem.__init__(self, **kwargs)

    def send(self,
             drop_duration: Optional[float] = None,
             data_bits: Optional[np.array] = None) -> np.ndarray:
        """Returns an array with the complex baseband samples of a waveform generator.

        The signal may be distorted by RF impairments.

        Args:
            drop_duration (float, optional): Length of signal in seconds.
            data_bits (np.array, optional): Data bits to be sent via this transmitter.

        Returns:
            np.ndarray:
                Complex baseband samples, rows denoting transmitter antennas and
                columns denoting samples.

        Raises:
            ValueError: If not enough data bits were provided to generate as single frame.
        """
        # coded_bits = self.encoder.encoder(data_bits)
        num_samples = int(np.ceil(drop_duration * self.scenario.sampling_rate))
        timestamps = np.arange(num_samples) / self.scenario.sampling_rate

        # Number of frames fitting into the selected drop duration
        frames_per_stream = int(ceil(drop_duration / self.waveform_generator.frame_duration))

        # Number of code bits required to generate all frames for all streams
        num_code_bits = self.waveform_generator.bits_per_frame * frames_per_stream * self.num_streams

        # Data bits required by the bit encoder to generate the input bits for the waveform generator
        num_data_bits = self.encoder_manager.required_num_data_bits(num_code_bits)

        # Generate source data bits if none are provided
        if data_bits is None:
            data_bits = self.__bits_source.get_bits(num_data_bits)[0]

        # Encode the data bits
        code_bits = self.encoder_manager.encode(data_bits, num_code_bits)

        # Map data bits to symbols
        symbols = self.waveform_generator.map(code_bits)

        # Apply symbol precoding
        symbol_streams = self.precoding.encode(symbols)

        # Generate a dedicated base-band signal for each symbol stream
        signal_streams = np.empty((symbol_streams.shape[0], num_samples), dtype=complex)

        for stream_idx, data_symbols in enumerate(symbol_streams):
            signal_streams[stream_idx, :] = self.waveform_generator.modulate(data_symbols, timestamps)

        # Apply stream coding, for instance beam-forming
        # TODO: Not yet supported.

        # Simulate the radio-frequency chain
        transmitted_signal = self.rf_chain.send(signal_streams)

        # Scale resulting signal by configured power factor
        transmitted_signal *= np.sqrt(self.power_factor)

        # We're finally done, blow the fanfares, throw confetti, etc.
        return transmitted_signal

        # Make sure enough data bits were provided
# elif len(data_bits) < num_data_bits:
#     raise ValueError("Number of provided data bits is insufficient to generate a single frame")

# # Apply channel coding to the source bits
# code_bits = self.encoder_manager.encode(data_bits, num_code_bits)

# while timestamp < number_of_samples:

#     # Generate base-band waveforms
#     frame, timestamp, initial_sample_num = self.waveform_generator.create_frame(
#         timestamp, code_bits)

#     if frame_index == 1:
#         tx_signal, samples_delay = self._allocate_drop_size(
#             initial_sample_num, number_of_samples)

#     tx_signal, samples_delay = self._add_frame_to_drop(
#         initial_sample_num, samples_delay, tx_signal, frame)
#     frame_index += 1

# # Create signal streams to each antenna via the precoder
# antenna_streams = self.precoding.encode(tx_signal)

# # Simulate RF chain
# tx_signal = self.rf_chain.send(antenna_streams)

# # Scale resulting signal by configured power factor
# tx_signal *= np.sqrt(self.power_factor)

# return tx_signal

    @classmethod
    def from_yaml(cls: Type[Transmitter], constructor: SafeConstructor, node: Node) -> Transmitter:
        """Recall a new `Transmitter` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Transmitter` serialization.

        Returns:
            Transmitter:
                Newly created `Transmitter` instance.

        Raises:
            RuntimeError: If `node` is neither a scalar or a map.
        """

        # If the transmitter is not a map, create a default object and warn the user
        if not isinstance(node, MappingNode):

            if isinstance(node, ScalarNode):
                return Transmitter()

            else:
                raise RuntimeError("Transmitters must be configured as YAML maps")

        constructor.add_multi_constructor(WaveformGenerator.yaml_tag, WaveformGenerator.from_yaml)
        state = constructor.construct_mapping(node, deep=True)

        bits_source = state.pop(BitsSource.yaml_tag, None)
        precoding = state.pop(SymbolPrecoding.yaml_tag, None)

        waveform_generator = None
        for key in state.keys():
            if key.startswith(WaveformGenerator.yaml_tag):
                waveform_generator = state.pop(key)
                break

        args = dict((k.lower(), v) for k, v in state.items())

        position = args.pop('position', None)
        orientation = args.pop('orientation', None)
        random_seed = args.pop('random_seed', None)

        if position is not None:
            args['position'] = np.array(position)

        if orientation is not None:
            args['orientation'] = np.array(orientation)

        # Convert the random seed to a new random generator object if its specified within the config
        if random_seed is not None:
            args['random_generator'] = rnd.default_rng(random_seed)

        transmitter = Transmitter(**args)

        if bits_source is not None:
            transmitter.bits_source = bits_source

        if precoding is not None:
            transmitter.precoding = precoding

        if waveform_generator is not None:
            transmitter.waveform_generator = waveform_generator

        return transmitter

    @property
    def index(self) -> int:
        """The index of this transmitter in the scenario.

        Returns:
            int:
                The index.
        """

        return self.scenario.transmitters.index(self)

    @property
    def paired_modems(self) -> List[Modem]:
        """The modems connected to this modem over an active channel.

        Returns:
            List[Modem]:
                A list of paired modems.
        """

        return [channel.receiver for channel in self.scenario.departing_channels(self, True)]

    def generate_data_bits(self) -> np.ndarray:
        """Generate data bits required to build a single transmit data frame for this modem.

        Returns:
            numpy.ndarray: A vector of hard data bits in 0/1 format.
        """

        return self.random_generator.integers(0, 2, self.num_data_bits_per_frame)

    @property
    def reference_channel(self) -> Channel:

        if self.scenario is None:
            raise RuntimeError("Attempting to access reference channel of a floating modem.")

        return self.scenario.departing_channels(self, active_only=True)[0]
