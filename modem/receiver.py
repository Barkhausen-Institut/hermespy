# -*- coding: utf-8 -*-
"""HermesPy Receiving Modem."""

from __future__ import annotations
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import TYPE_CHECKING, Type, List, Optional
from math import ceil
import numpy as np
import numpy.random as rnd

from source import BitsSource
from modem import Modem
from modem.precoding import SymbolPrecoding
from modem.waveform_generator import WaveformGenerator
from noise import Noise

if TYPE_CHECKING:
    from .transmitter import Transmitter
    from channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Receiver(Modem):
    """Receiving modem within a scenario configuration.

    Attributes:

        __noise: The noise model.
    """

    yaml_tag = 'Receiver'
    __noise: Noise
    __reference_transmitter: Optional[Transmitter]

    def __init__(self, **kwargs) -> None:
        """Receiver modem object initialization.

        Args:

            noise(Noise, optional): Noise generator.
        """

        noise = kwargs.pop('noise', None)

        Modem.__init__(self, **kwargs)

        self.noise = Noise() if noise is None else noise
        self.reference_transmitter = None

    def receive(self, input_signals: np.ndarray, noise_power: float) -> np.ndarray:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            input_signals (np.ndarray): Received signal.
            noise_power (float): Power of the incoming noise, for simulation and possible equalization.

        Returns:
            np.array: Detected bits as a list of data blocks for the drop.

        Raises:
            ValueError: If the first dimension of `input_signals` does not match the number of receive antennas.
        """

        if input_signals.shape[0] != self.num_antennas:
            raise ValueError("Number of input signals must be equal to the number of antennas")

        # If no receiving waveform generator is configured, no signal is being received
        # TODO: Check if this is really a valid case
        if self.waveform_generator is None:
            return np.empty((self.num_antennas, 0), dtype=complex)

        num_samples = input_signals.shape[1]
        timestamps = np.arange(num_samples) / self.scenario.sampling_rate

        # Number of frames within the received samples
        frames_per_stream = int(ceil(num_samples / self.waveform_generator.samples_in_frame))

        # Number of simples pre received stream
        symbols_per_stream = frames_per_stream * self.waveform_generator.symbols_per_frame

        # Number of code bits required to generate all frames for all streams
        num_code_bits = self.waveform_generator.bits_per_frame * frames_per_stream * self.num_streams

        # Data bits required by the bit encoder to generate the input bits for the waveform generator
        num_data_bits = self.encoder_manager.required_num_data_bits(num_code_bits)

        noise_var = noise_power / 1.0  # TODO: Re-implement pair power factor

        # Add receive noise
        noisy_signals = self.__noise.add_noise(input_signals, noise_var)

        # Simulate the radio-frequency chain
        received_signals = self.rf_chain.receive(noisy_signals)

        # Scale resulting signal by configured power factor
        received_signals /= np.sqrt(self.power_factor)  # TODO: Re-implement pair power factor

        # Apply stream decoding, for instance beam-forming
        # TODO: Not yet supported.

        # Generate a symbol stream for each dedicated base-band signal
        symbol_streams = np.empty((received_signals.shape[0], symbols_per_stream),
                                  dtype=self.waveform_generator.symbol_type)

        for stream_idx, noisy_signal in enumerate(noisy_signals):
            symbol_streams[stream_idx, :] = self.waveform_generator.demodulate(noisy_signal, timestamps)

        # Decode the symbol precoding
        symbols = self.precoding.decode(symbol_streams)

        # Map the symbols to code bits
        code_bits = self.waveform_generator.unmap(symbols)

        # Decode the coded bit stream to plain data bits
        data_bits = self.encoder_manager.decode(code_bits, num_data_bits)

        # We're finally done, blow the fanfares, throw confetti, etc.
        return data_bits

    @classmethod
    def from_yaml(cls: Type[Receiver], constructor: RoundTripConstructor, node: Node) -> Receiver:

        state = constructor.construct_mapping(node, CommentedOrderedMap)

        waveform_generator = None
        bits_source = None
        precoding = state.pop(SymbolPrecoding.yaml_tag, None)

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
        orientation = args.pop('orientation', None)
        random_seed = args.pop('random_seed', None)
        noise = args.pop('noise', None)

        if position is not None:
            args['position'] = np.array(position)

        if orientation is not None:
            args['orientation'] = np.array(orientation)

        # Convert the random seed to a new random generator object if its specified within the config
        if random_seed is not None:
            args['random_generator'] = rnd.default_rng(random_seed)

        # Create new receiver object
        receiver = Receiver(**args)

        # Update noise model if specified by the configuration
        if noise is not None:
            receiver.noise = noise

        if precoding is not None:
            receiver.precoding = precoding

        return receiver

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

    @property
    def noise(self) -> Noise:
        """Access this receiver's noise model configuration.

        Returns:
            Noise: Handle the noise model.
        """

        return self.__noise

    @noise.setter
    def noise(self, model: Noise) -> None:
        """Modify this receiver's noise model configuration.

        Args:
            model (Noise): The new noise model instance.

        Raises:
            RuntimeError: If the `model` is already attached to a different receiver.
        """

        self.__noise = model
        model.receiver = self

    @property
    def reference_transmitter(self) -> Optional[Transmitter]:
        """Reference modem transmitting to this receiver.

        Used to for channel estimation, noise estimation, power configuration etc.

        Return:
            Optional[Transmitter]: Transmitting reference modem. None if no reference is configured.
        """

        return self.__reference_transmitter

    @reference_transmitter.setter
    def reference_transmitter(self, new_reference: Optional[Transmitter]) -> None:
        """Modify reference modem transmitting to this receiver.

        Args:
            Optional[Transmitter]: Transmitting reference modem.

        Raises:

            RuntimeError:
                If this Receiver is currently floating.

            ValueError:
                If `new_reference` is not registered with the Receiver's scenario.
        """

        if new_reference is None:

            self.__reference_transmitter = None

        else:

            if self.scenario is None:
                raise RuntimeError("Error trying to modify the reference transmitter of a floating receiver")

            if new_reference not in self.scenario.transmitters:
                raise ValueError("Error trying to configure a reference transmitter not within a receiver's scenario")

            self.__reference_transmitter = new_reference

    @property
    def reference_channel(self) -> Channel:

        if self.scenario is None:
            raise RuntimeError("Attempting to access reference channel of a floating modem.")

        # If no reference transmitter is configured, guess the paired modem's channel as the diagonal pair in the
        # channel matrix
        if self.__reference_transmitter is None:

            guessed_pair_index = min(self.scenario.num_receivers-1, self.index)
            return self.scenario.arriving_channels(self)[guessed_pair_index]

        else:

            return self.scenario.channel(self.__reference_transmitter, self)
