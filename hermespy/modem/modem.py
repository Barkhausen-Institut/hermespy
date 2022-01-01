# -*- coding: utf-8 -*-
"""
=====
Modem
=====
"""

from __future__ import annotations
from typing import Any, List, Tuple, Type, Optional
from math import floor

import numpy as np
from ruamel.yaml import SafeRepresenter, SafeConstructor, MappingNode

from hermespy.channel import ChannelStateDimension, ChannelStateInformation
from hermespy.coding import EncoderManager
from hermespy.core import DuplexOperator, RandomNode
from hermespy.core.factory import SerializableArray
from hermespy.core.signal_model import Signal
from hermespy.precoding import SymbolPrecoding
from .bits_source import BitsSource, RandomBitsSource
from .waveform_generator import WaveformGenerator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Modem(RandomNode, DuplexOperator, SerializableArray):
    """HermesPy representation of a wireless communication modem.

    Modems may transmit or receive information in form of bit streams.

    In HermesPy, a modem is the basis of every simulation entity which may transmit or receive
    electromagnetic waveforms.

    The modem consists of an analog RF chain, a waveform generator, and can be used
    either for transmission or reception of a given technology.
    """

    yaml_tag = u'Modem'
    """YAML serialization tag."""

    __encoder_manager: EncoderManager
    __precoding: SymbolPrecoding
    __waveform_generator: Optional[WaveformGenerator]
    __bits_source: BitsSource
    __transmitted_bits: np.ndarray                      # Cache of recently transmitted bits
    __transmitted_symbols: np.ndarray                   # Cache of recently transmitted symbols
    __received_bits: np.ndarray                         # Cache of recently received bits
    __received_symbols: np.ndarray                      # Cache of recently received symbols

    def __init__(self,
                 encoding: Optional[EncoderManager] = None,
                 precoding: Optional[SymbolPrecoding] = None,
                 waveform: Optional[WaveformGenerator] = None,
                 seed: Optional[int] = None) -> None:
        """
        Args:

            encoding (EncoderManager, optional):
                Bit coding configuration.
                Encodes communication bit frames during transmission and decodes them during reception.

            precoding (SymbolPrecoding, optional):
                Modulation symbol coding configuration.

            waveform (WaveformGenerator, optional):
                The waveform to be transmitted by this modem.

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.
        """

        # Base class initialization
        RandomNode.__init__(self, seed=seed)
        DuplexOperator.__init__(self)

        self.__encoder_manager = EncoderManager()
        self.__precoding = SymbolPrecoding(modem=self)
        self.__waveform_generator = None
        self.__power = 1.0
        self.__transmitted_bits = np.empty(0, dtype=int)
        self.__transmitted_symbols = np.empty(0, dtype=complex)
        self.__received_bits = np.empty(0, dtype=int)
        self.__received_symbols = np.empty(0, dtype=complex)

        self.bits_source = RandomBitsSource()
        self.encoder_manager = EncoderManager() if encoding is None else encoding
        self.precoding = SymbolPrecoding(modem=self) if precoding is None else precoding
        self.waveform_generator = waveform

    def transmit(self,
                 duration: float = 0.) -> Tuple[Signal, np.ndarray, np.ndarray]:
        """Returns an array with the complex base-band samples of a waveform generator.

        The signal may be distorted by RF impairments.

        Args:
            duration (float, optional): Length of signal in seconds.

        Returns:
            transmissions (tuple):

                signal (Signal):
                    Signal model carrying the `data_bits` in multiple streams, each stream encoding multiple
                    radio frequency band communication frames.

                data_symbols (np.ndarray):
                    Vector of symbols to which `data_bits` were mapped, used to modulate `signal`.

                data_bits (np.ndarray):
                    Vector of bits mapped to `data_symbols`.
        """

        # By default, the drop duration will be exactly one frame
        if duration <= 0.:
            duration = self.frame_duration

        # Number of data symbols per transmitted frame
        symbols_per_frame = self.waveform_generator.symbols_per_frame

        # Number of frames fitting into the selected drop duration
        frames_per_stream = int(duration / self.waveform_generator.frame_duration)

        # Generate data bits
        data_bits = self.generate_data_bits()

        # Number of code bits required to generate all frames for all streams
        num_code_bits = int(self.waveform_generator.bits_per_frame * frames_per_stream / self.precoding.rate)

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
        signal = Signal(np.empty((0, 0), dtype=complex),
                        self.waveform_generator.sampling_rate,
                        self.device.carrier_frequency)

        for stream_idx, stream_symbols in enumerate(symbol_streams):

            stream_signal = Signal(np.empty((0, 0), dtype=complex),
                                   self.waveform_generator.sampling_rate,
                                   self.device.carrier_frequency)

            for frame_idx in range(frames_per_stream):

                data_symbols = stream_symbols[frame_idx*symbols_per_frame:(1+frame_idx)*symbols_per_frame]

                frame_signal = self.waveform_generator.modulate(data_symbols)
                stream_signal.append_samples(frame_signal)

            signal.append_streams(stream_signal)

        # Apply stream coding, for instance beam-forming
        # TODO: Not yet supported.

        # Transmit signal over the occupied device slot (if the modem is attached to a device)
        if self._transmitter.attached:
            self._transmitter.slot.add_transmission(self._transmitter, signal)

        # Simulate the radio-frequency chain
        # signal.samples = self.rf_chain.send(signal.samples)

        # Scale resulting signal by configured power factor
        # signal.samples *= np.sqrt(self.power)

        # Cache transmissions
        self.__transmitted_bits = data_bits
        self.__transmitted_symbols = symbols

        # We're finally done, blow the fanfares, throw confetti, etc.
        return signal, symbols, data_bits
    
    @property
    def transmitted_bits(self) -> np.ndarray:
        """Recently transmitted data bits.
        
        Returns:
            np.ndarray: Numpy array of recently transmitted data bits.
        """
        
        return self.__transmitted_bits.copy()
    
    @property
    def transmitted_symbols(self) -> np.ndarray:
        """Recently transmitted modulation symbols.
        
        Returns:
            np.ndarray: Numpy array of recently transmitted symbols.
        """
        
        return self.__transmitted_symbols.copy()

    def receive(self) -> Tuple[Any, ...]:

        signal = self._receiver.signal
        if signal is None:
            raise RuntimeError("No signal received by modem")
            # signal = Signal.empty(sampling_rate=self.device.sampling_rate)

        csi = self._receiver.csi
        if csi is None:
            csi = ChannelStateInformation.Ideal(signal.num_samples)

        # Pull signal and channel state from the registered device slot
        noise = 0.  # ToDo: Re-implement noise

        num_samples = signal.num_samples

        # Number of frames within the received samples
        frames_per_stream = int(floor(num_samples / self.waveform_generator.samples_in_frame))

        # Number of code bits required to generate all frames for all streams
        num_code_bits = int(self.waveform_generator.bits_per_frame * frames_per_stream / self.precoding.rate)

        # Data bits required by the bit encoder to generate the input bits for the waveform generator
        num_data_bits = self.encoder_manager.required_num_data_bits(num_code_bits)

        # Apply stream decoding, for instance beam-forming
        # TODO: Not yet supported.

        # Synchronize all streams into frames
        frames: List[List[Tuple[np.ndarray, ChannelStateInformation]]] = []
        for stream_idx, (rx_signal, stream_transform) in enumerate(zip(signal.samples,
                                                                       csi.received_streams())):

            frame = self.waveform_generator.synchronization.synchronize(rx_signal, stream_transform)
            frames.append(frame)

        frames = np.array(frames, dtype=object)

        # Abort at this point if no frames have been detected
        if frames.size < 1:
            return signal, np.empty(0, dtype=complex), np.empty(0, dtype=complex)

        # Demodulate the parallel frames arriving at each stream,
        # then decode the (inverse) precoding over all stream frames
        decoded_symbols = np.empty(0, dtype=complex)
        for frame_streams in frames.transpose(1, 0, 2):

            # Demodulate each frame separately
            symbols: List[np.ndarray] = []
            channel_states: List[ChannelStateInformation] = []
            noises: List[np.ndarray] = []

            for stream in frame_streams:

                # Demodulate the frame into data symbols
                s_symbols, s_channel_state, s_noise = self.waveform_generator.demodulate(*stream, noise)

                symbols.append(s_symbols)
                channel_states.append(s_channel_state)
                noises.append(s_noise)

            frame_symbols = np.array(symbols, dtype=complex)
            frame_channel_states = ChannelStateInformation.concatenate(channel_states,
                                                                       ChannelStateDimension.RECEIVE_STREAMS)
            frame_noises = np.array(noises, dtype=float)

            decoded_frame_symbols = self.precoding.decode(frame_symbols,
                                                          frame_channel_states,
                                                          frame_noises)
            decoded_symbols = np.append(decoded_symbols, decoded_frame_symbols)

        # Map the symbols to code bits
        code_bits = self.waveform_generator.unmap(decoded_symbols)

        # Decode the coded bit stream to plain data bits
        data_bits = self.encoder_manager.decode(code_bits, num_data_bits)

        # Cache receptions
        self.__received_bits = data_bits
        self.__received_symbols = decoded_symbols

        # We're finally done, blow the fanfares, throw confetti, etc.
        return signal, decoded_symbols, data_bits

    @property
    def received_bits(self) -> np.ndarray:
        """Recently received data bits.

        Returns:
            np.ndarray: Numpy array of recently received data bits.
        """

        return self.__received_bits.copy()

    @property
    def received_symbols(self) -> np.ndarray:
        """Recently received modulation symbols.

        Returns:
            np.ndarray: Numpy array of recently received symbols.
        """

        return self.__received_symbols.copy()

    @property
    def num_streams(self) -> int:
        """The number of data streams handled by the modem.

        The number of data streams is always less or equal to the number of available antennas `num_antennas`.

        Returns:
            int:
                The number of data streams generated by the modem.
        """

        # For now, stream compression will not be supported
        return self.device.num_antennas

    @property
    def bits_source(self) -> BitsSource:
        """Source of bits transmitted over the modem.

        Returns:
            bits_source (BitsSource): Handle to the bits source.
        """

        return self.__bits_source

    @bits_source.setter
    def bits_source(self, value: BitsSource) -> None:
        """Set the source of bits transmitted over the modem"""

        self.__bits_source = value
        self.__bits_source.random_mother = self

    @property
    def encoder_manager(self) -> EncoderManager:
        """Access the modem's encoder management.

        Returns:
            EncoderManager:
                Handle to the modem's encoder instance.
        """

        return self.__encoder_manager

    @encoder_manager.setter
    def encoder_manager(self, new_manager: EncoderManager) -> None:
        """Update the modem's encoder management.

        Args:
            new_manager (EncoderManger):
                The new encoder manager.
        """

        self.__encoder_manager = new_manager
        new_manager.modem = self

    @property
    def waveform_generator(self) -> WaveformGenerator:
        """Communication waveform emitted by this modem.

        Returns:
            WaveformGenerator:
                Handle to the modem's `WaveformGenerator` instance.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, value: Optional[WaveformGenerator]) -> None:
        """Set the communication waveform emitted by this modem."""

        self.__waveform_generator = value

        if value is not None:
            value.modem = self

    @property
    def precoding(self) -> SymbolPrecoding:
        """Access this modem's precoding configuration.

        Returns:
            SymbolPrecoding: Handle to the configuration.
        """

        return self.__precoding

    @precoding.setter
    def precoding(self, coding: SymbolPrecoding) -> None:
        """Modify the modem's precoding configuration.

        Args:
            coding (SymbolPrecoding): The new precoding configuration.
        """

        self.__precoding = coding
        self.__precoding.modem = self

    @property
    def num_data_bits_per_frame(self) -> int:
        """Compute the number of required data bits to generate a single frame.

        Returns:
            int: The number of data bits.
        """

        num_code_bits = self.waveform_generator.bits_per_frame
        return self.encoder_manager.required_num_data_bits(num_code_bits)

    @property
    def frame_duration(self) -> float:

        return self.waveform_generator.frame_duration

    @property
    def sampling_rate(self) -> float:

        return self.waveform_generator.sampling_rate

    @property
    def energy(self) -> float:

        if self.waveform_generator is None:
            return 0.

        return self.waveform_generator.bit_energy

    def generate_data_bits(self) -> np.ndarray:
        """Generate data bits required to build a single transmit data frame for this modem.

        Returns:
            numpy.ndarray: A vector of hard data bits in 0/1 format.
        """

        num_bits = int(self.num_data_bits_per_frame / self.precoding.rate)
        bits = self.bits_source.generate_bits(num_bits)
        return bits

    @classmethod
    def to_yaml(cls: Type[Modem], representer: SafeRepresenter, node: Modem) -> MappingNode:
        """Serialize a `Modem` object to YAML.

        Args:

            representer (Modem):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Modem):
                The `Device` instance to be serialized.

        Returns:

            MappingNode:
                The serialized YAML node.
        """

        state = {}

        if len(node.__encoder_manager.encoders) > 0:
            state['Encoding'] = node.__encoder_manager

        if len(node.__precoding) > 0:
            state['Precoding'] = node.__precoding

        if node.__waveform_generator is not None:
            state['Waveform'] = node.__waveform_generator

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[Modem], constructor: SafeConstructor, node: MappingNode) -> Modem:
        """Recall a new `Modem` class instance from YAML.

        Args:

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (MappingNode):
                YAML node representing the `Modem` serialization.

        Returns:

            Modem:
                Newly created serializable instance.
        """

        state = constructor.construct_mapping(node)

        encoding: Optional[EncoderManager] = state.pop('Encoding', None)
        precoding: Optional[SymbolPrecoding] = state.pop('Precoding', None)
        waveform: Optional[WaveformGenerator] = state.pop('Waveform', None)

        modem = cls(**state)

        if encoding is not None:
            modem.encoder_manager = encoding

        if precoding is not None:
            modem.precoding = precoding

        if waveform is not None:
            modem.waveform_generator = waveform

        return modem
