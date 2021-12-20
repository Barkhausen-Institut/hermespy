# -*- coding: utf-8 -*-
"""
=====
Modem
=====
"""

from __future__ import annotations
from typing import Any, Tuple, Type, Optional

import numpy as np
from ruamel.yaml import SafeRepresenter, MappingNode

from hermespy.coding import EncoderManager
from hermespy.core import Device, DuplexOperator, RandomNode
from hermespy.precoding import SymbolPrecoding
from hermespy.signal import Signal
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


class Modem(RandomNode, DuplexOperator):
    """HermesPy representation of a wireless communication modem.

    Modems may transmit or receive information in form of bit streams.

    In HermesPy, a modem is the basis of every simulation entity which may transmit or receive
    electromagnetic waveforms.

    The modem consists of an analog RF chain, a waveform generator, and can be used
    either for transmission or reception of a given technology.
    """

    __slots__ = ['__encoder_manager', '__waveform_generator', '__bits_source']

    yaml_tag = u'Modem'
    """YAML serialization tag."""

    __encoder_manager: EncoderManager
    __precoding: SymbolPrecoding
    __waveform_generator: Optional[WaveformGenerator]
    __bits_source: BitsSource

    def __init__(self,
                 encoding: Optional[EncoderManager] = None,
                 precoding: Optional[SymbolPrecoding] = None,
                 waveform: Optional[WaveformGenerator] = None,
                 seed: Optional[int] = None,
                 *args: Any,
                 **kwargs: Any) -> None:
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

            *args (Any):
                Operator base class initialization parameters.

            **kwargs (Any):
                Operator base class initialization parameters.
        """

        # Base class initialization
        RandomNode.__init__(self, seed=seed)
        DuplexOperator.__init__(self)

        self.__carrier_frequency = 800e6
        self.__encoder_manager = EncoderManager()
        self.__precoding = SymbolPrecoding(modem=self)
        self.__waveform_generator = None
        self.__power = 1.0

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

        # We're finally done, blow the fanfares, throw confetti, etc.
        return signal, symbols, data_bits

#    def receive(self, rf_signals: Union[List[Signal], Signal],
#                noise_variance: float = 0.0) -> Signal:
#        """Receive and down-mix radio-frequency-band signals to base-band signals over the receiver's hardware chain.
#
#        Args:
#
#            rf_signals (Union[List[Signal], Signal]):
#                List containing radio-frequency band signal models impinging onto the receiver
#
#            noise_variance (float, optional):
#                Variance (i.e. power) of the thermal noise added to the signals during reception.
#
#        Returns:
#
#            Signal:
#                A superposition of all `rf_signals` mixed to the base-band, distorted by noise and
#                hardware-effects.
#
#        Raises:
#
#            ValueError:
#                If the first dimension of each rf signal does not match the number of receive antennas.
#                If `noise_variance` is smaller than zero.
#        """
#
#        baseband_signal = Signal.empty(self.waveform_generator.sampling_rate,
#                                       carrier_frequency=self.carrier_frequency,
#                                       num_streams=self.num_antennas)
#
#        # Down-mix each rf-band signal to base-band and superimpose them at the receiver-side
#        if isinstance(rf_signals, Signal):
#            baseband_signal.superimpose(rf_signals)
#
#        else:
#            for rf_signal in rf_signals:
#                baseband_signal.superimpose(rf_signal)
#
#        # Scale resulting signal to unit power (relative to the configured transmitter reference)
#        baseband_signal.samples /= np.sqrt(self.received_power)
#
#        # Add receive noise
#        noisy_signal = baseband_signal.copy()
#        noisy_signal.samples = self.__noise.add_noise(baseband_signal.samples, noise_variance)
#
#        # Simulate the radio-frequency chain
#        received_signal = noisy_signal.copy()
#        received_signal.samples = self.rf_chain.receive(received_signal.samples)
#
#        # Return resulting base-band superposition
#        return received_signal

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
        """Serialize a modem object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Modem):
                The modem instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        serialization = {
            "carrier_frequency": node.__carrier_frequency,
            "tx_power": node.__power,
            EncoderManager.yaml_tag: node.__encoder_manager,
            SymbolPrecoding.yaml_tag: node.__precoding,
        }

        if node.waveform_generator is not None:
            serialization[node.waveform_generator.yaml_tag] = node.waveform_generator

        return representer.represent_mapping(cls.yaml_tag, serialization)
