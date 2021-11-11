# -*- coding: utf-8 -*-
"""HermesPy transmitting modem."""

from __future__ import annotations
from ruamel.yaml import SafeConstructor, Node, MappingNode, ScalarNode
from typing import Type, List, Any, Optional
import numpy as np
import numpy.random as rnd

from hermespy.modem import Modem
from hermespy.source import BitsSource
from hermespy.modem.waveform_generator import WaveformGenerator
from hermespy.modem.precoding import SymbolPrecoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Transmitter(Modem):
    """Transmitting modem within a scenario configuration.

    Attributes:

        __power (float):
            Mean transmission power in Watts.

        bits_source (BitsSource):
            Source of bits to be transmitted.
    """

    yaml_tag = 'Transmitter'

    __power: float
    bits_source: BitsSource

    def __init__(self,
                 power: float = 1.0,
                 bits_source: Optional[BitsSource] = None,
                 **kwargs: Any) -> None:
        """Object initialization.

        Args:

            power (float, optional):
                Average power of the transmitted signal. 1.0 By default.

            bits_source (BitsSource, optional):
                Source of bits to be transmitted.

            **kwargs (Any):
                Modem configuration parameters.
        """

        # Init base class
        Modem.__init__(self, **kwargs)

        # Init parameters
        self.power = power
        self.bits_source = BitsSource(self) if bits_source is None else bits_source

    @property
    def power(self) -> float:
        """Power of the transmitted signal.

        Returns:
            float: Transmit power in Watt.
        """

        return self.__power

    @power.setter
    def power(self, new_power: float) -> None:
        """Modify the power of the transmitted signal.

        Args:
            new_power (float): The new signal transmit power in Watt.

        Raises:
            ValueError: If transmit power is negative.
        """

        if new_power < 0.0:
            raise ValueError("Transmit power must be greater or equal to zero")

        self.__power = new_power

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

        # By default, the drop duration will be exactly one frame
        if drop_duration is None:
            drop_duration = self.waveform_generator.frame_duration

        # coded_bits = self.encoder.encoder(data_bits)
        num_samples = int(np.ceil(drop_duration * self.scenario.sampling_rate))
        timestamps = np.arange(num_samples) / self.scenario.sampling_rate

        # Number of data symbols per transmitted frame
        symbols_per_frame = self.waveform_generator.symbols_per_frame

        # Length of frame in samples
        samples_per_frame = self.waveform_generator.samples_in_frame

        # Number of frames fitting into the selected drop duration
        frames_per_stream = int(np.ceil(drop_duration / self.waveform_generator.frame_duration))

        # Number of code bits required to generate all frames for all streams
        num_code_bits = int(self.waveform_generator.bits_per_frame * frames_per_stream / self.precoding.rate)

        # Generate source data bits if none are provided
        if data_bits is None:
            data_bits = self.generate_data_bits()

        # Encode the data bits
        code_bits = self.encoder_manager.encode(data_bits, num_code_bits)

        # Map data bits to symbols
        symbols = self.waveform_generator.map(code_bits)

        # Apply symbol precoding
        symbol_streams = self.precoding.encode(symbols)

        # Check that the number of symbol streams matches the number of required symbol streams
        if symbol_streams.shape[0] != self.num_streams:
            raise RuntimeError("Invalid precoding configuration, the number of resulting streams does not "
                               "match the number of transmit antennas")

        # Generate a dedicated base-band signal for each symbol stream
        signal_streams = np.empty((symbol_streams.shape[0], num_samples), dtype=complex)

        for stream_idx, stream_symbols in enumerate(symbol_streams):
            for frame_idx in range(frames_per_stream):

                data_symbols = stream_symbols[frame_idx*symbols_per_frame:(1+frame_idx)*symbols_per_frame]
                frame = self.waveform_generator.modulate(data_symbols, timestamps)

                frame_start_idx = min(num_samples, frame_idx * samples_per_frame)
                frame_end_idx = min(num_samples, (1 + frame_idx) * samples_per_frame)
                frame_length = frame_end_idx - frame_start_idx

                signal_streams[stream_idx, frame_start_idx:frame_end_idx] = frame[:frame_length]

        # Apply stream coding, for instance beam-forming
        # TODO: Not yet supported.

        # Simulate the radio-frequency chain
        transmitted_signal = self.rf_chain.send(signal_streams)

        # Scale resulting signal by configured power factor
        transmitted_signal *= np.sqrt(self.power)

        # We're finally done, blow the fanfares, throw confetti, etc.
        return transmitted_signal

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

        num_bits = int(self.num_data_bits_per_frame * self.precoding.rate)
        bits = self.bits_source.get_bits(num_bits)
        return bits
