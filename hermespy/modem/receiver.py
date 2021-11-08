# -*- coding: utf-8 -*-
"""HermesPy Receiving Modem."""

from __future__ import annotations
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import TYPE_CHECKING, Type, List, Optional, Union, Tuple
from math import floor
from scipy.constants import pi
import numpy as np
import numpy.random as rnd

from modem import Modem
from modem.precoding import SymbolPrecoding
from modem.waveform_generator import WaveformGenerator
from noise import Noise

if TYPE_CHECKING:
    from scenario import Scenario
    from .transmitter import Transmitter
    from channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Receiver(Modem):
    """Receiving modem within a scenario configuration.

    Attributes:

        __noise:
            The noise model for thermal effects during reception sampling.

        __reference_transmitter:
            Referenced transmit peer for this receiver.
    """

    yaml_tag = 'Receiver'
    __noise: Noise
    __reference_transmitter: Union[int, Transmitter, None]

    def __init__(self, **kwargs) -> None:
        """Receiver modem object initialization.

        Args:

            noise(Noise, optional): Noise generator.
        """

        noise = kwargs.pop('noise', None)
        reference_transmitter = kwargs.pop('reference_transmitter', None)

        Modem.__init__(self, **kwargs)

        self.noise = Noise() if noise is None else noise
        self.reference_transmitter = reference_transmitter

    def receive(self, rf_signals: List[Tuple[np.ndarray, float]],
                noise_variance: float) -> np.ndarray:
        """Receive and down-mix radio-frequency-band signals to baseband signals over the receiver's hardware chain.

        Args:
            rf_signals (List[Tuple[np.ndarray, float]]):
                List containing pairs of rf-band samples and their respective carrier frequencies.

            noise_variance (float):
                Variance (i.e. power) of the thermal noise added to the signals during reception.

            Raises:
                ValueError:
                    If the first dimension of each rf signal does not match the number of receive antennas.
                    If `noise_variance` is smaller than zero.
        """

        max_signal_length = 0
        for rf_signal, _ in rf_signals:

            # Make sure the rf signal contains a stream for each antenna
            if rf_signal.shape[0] != self.num_antennas:
                raise ValueError("Number of input signal streams must be equal to the number of receiving antennas")

            max_signal_length = max(max_signal_length, rf_signal.shape[1])

        timestamps = 2j * pi * np.arange(max_signal_length) / self.scenario.sampling_rate
        baseband_signal = np.zeros((self.num_antennas, max_signal_length), dtype=complex)

        # Down-mix each rf-band signal to baseband and superimpose them at the receiver-side
        for rf_signal, carrier_frequency in rf_signals:

            frequency_difference = carrier_frequency - self.carrier_frequency
            mix_oscillation = np.exp(timestamps[:rf_signal.shape[1]] * frequency_difference)

            for antenna_idx, rf_antenna_signal in enumerate(rf_signal):
                baseband_signal[antenna_idx, :rf_signal.shape[1]] += mix_oscillation * rf_antenna_signal

        # Scale resulting signal to unit power (relative to the configured transmitter reference)
        scaled_signal = baseband_signal / np.sqrt(self.received_power)

        # Add receive noise
        noisy_signal = self.__noise.add_noise(scaled_signal, noise_variance)

        # Simulate the radio-frequency chain
        received_signal = self.rf_chain.receive(noisy_signal)

        # Return resulting base-band superposition
        return received_signal

    def demodulate(self, baseband_signal: np.ndarray, channel: np.ndarray, noise_variance: float) -> np.ndarray:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            baseband_signal (np.ndarray):
                Received signal noisy base-band signal.

            channel (np.ndarray):
                Wireless channel impulse-response estimation.

            noise_variance (float):
                Power of the incoming noise, required for equalization and channel estimation.

        Returns:
            np.array: Detected bits as a list of data blocks for the drop.

        Raises:
            ValueError: If the first dimension of `input_signals` does not match the number of receive antennas.
        """

        if baseband_signal.shape[0] != self.num_antennas:
            raise ValueError("Number of input signals must be equal to the number of antennas")

        # If no receiving waveform generator is configured, no signal is being received
        # TODO: Check if this is really a valid case
        if self.waveform_generator is None:
            return np.empty((self.num_antennas, 0), dtype=complex)

        num_samples = baseband_signal.shape[1]

        # Number of frames within the received samples
        frames_per_stream = int(floor(num_samples / self.waveform_generator.samples_in_frame))

        # Number of code bits required to generate all frames for all streams
        num_code_bits = int(self.waveform_generator.bits_per_frame * frames_per_stream * self.precoding.rate)

        # Data bits required by the bit encoder to generate the input bits for the waveform generator
        num_data_bits = self.encoder_manager.required_num_data_bits(num_code_bits)

        # Apply stream decoding, for instance beam-forming
        # TODO: Not yet supported.

        # Since no spatial stream coding is supported,
        # the channel response at each transmit input is the sum over all impinging antenna signals
        # ToDo: This is probably not correct, since it depends on the multiplexing
        stream_responses = np.sum(channel, axis=2)

        # Generate a symbol stream for each dedicated base-band signal
        symbol_streams: List[List[complex]] = []
        symbol_streams_responses: List[List[complex]] = []
        symbol_streams_noise: List[List[float]] = []

        for stream_idx, (rx_signal, stream_response) in enumerate(zip(baseband_signal,
                                                                      np.rollaxis(stream_responses, 1))):

            # Synchronization
            frame_samples, frame_responses = self.waveform_generator.synchronize(rx_signal, stream_response)

            # Demodulate each frame separately to make the de-modulation easier to understand
            symbols: List[complex] = []
            symbol_responses: List[complex] = []
            symbol_noise: List[float] = []
            for frame, response in zip(frame_samples, frame_responses):

                # Demodulate the frame into dat symbols
                f_symbols, f_responses, f_noise = self.waveform_generator.demodulate(frame, response, noise_variance)

                symbols.extend(f_symbols.tolist())
                symbol_responses.extend(f_responses.tolist())
                symbol_noise.extend(f_noise.tolist())

            # Save data symbols in their respective stream section
            symbol_streams.append(symbols)
            symbol_streams_responses.append(symbol_responses)
            symbol_streams_noise.append(symbol_noise)

        # Decode the symbol precoding
        decoded_symbols = self.precoding.decode(np.array(symbol_streams),
                                                np.array(symbol_streams_responses),
                                                np.array(symbol_streams_noise))

        # Map the symbols to code bits
        code_bits = self.waveform_generator.unmap(decoded_symbols)

        # Decode the coded bit stream to plain data bits
        data_bits = self.encoder_manager.decode(code_bits, num_data_bits)

        # We're finally done, blow the fanfares, throw confetti, etc.
        return data_bits

    @classmethod
    def from_yaml(cls: Type[Receiver], constructor: RoundTripConstructor, node: Node) -> Receiver:

        state = constructor.construct_mapping(node, CommentedOrderedMap)

        waveform_generator = None
        precoding = state.pop(SymbolPrecoding.yaml_tag, None)

        for key in state.keys():
            if key.startswith(WaveformGenerator.yaml_tag):
                waveform_generator = state.pop(key)
                break

        state[WaveformGenerator.yaml_tag] = waveform_generator

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
        By default, returns the transmitter with the same index as this transmitter,
        i.e. searches along the diagonal channel matrix.

        Return:
            Optional[Transmitter]:
                Transmitting reference modem.
                None if the scenario contains no transmitter.

        Raises:
            RuntimeError: If the receiver is currently floating.
        """

        if self.__reference_transmitter is None:

            if self.scenario.num_transmitters < 1:
                return None

            else:
                guessed_pair_index = min(self.scenario.num_receivers-1, self.index)
                return self.scenario.transmitters[guessed_pair_index]

        else:
            return self.__reference_transmitter

    @reference_transmitter.setter
    def reference_transmitter(self, new_reference: Union[Transmitter, int, None]) -> None:
        """Modify reference modem transmitting to this receiver.

        Args:
            new_reference (Union[Transmitter, int, None]):
                Transmitting reference modem.
                May be a direct handle to the receiver or the receivers ID.
                None to remove the reference.

        Raises:

            ValueError:
                If `new_reference` is not registered with the Receiver's scenario.
        """

        if new_reference is None:

            self.__reference_transmitter = None

        else:

            if not self.is_attached:
                self.__reference_transmitter = new_reference

            else:

                # Convert reference IDs to reference handles
                if isinstance(new_reference, int):

                    if new_reference >= self.scenario.num_transmitters or new_reference < 0:
                        raise ValueError("Error trying to configure a reference transmitter ID not "
                                         "within a receiver's scenario")

                    # Convert the ID to an actual reference
                    new_reference = self.scenario.transmitters[new_reference]

                # Check if the handles are actually  transmitters within the scenario
                if new_reference not in self.scenario.transmitters:
                    raise ValueError("Error trying to configure a reference transmitter not "
                                     "within a receiver's scenario")

            self.__reference_transmitter = new_reference

    @Modem.scenario.setter
    def scenario(self, scenario: Scenario) -> None:

        # Call base class property setter
        Modem.scenario.fset(self, scenario)

        # Update the reference transmitter, implicitly runs a check if the reference transmitter
        # is contained within the set scenario
        if self.__reference_transmitter is not None:
            self.reference_transmitter = self.__reference_transmitter

    @property
    def reference_channel(self) -> Channel:

        if self.scenario is None:
            raise RuntimeError("Attempting to access reference channel of a floating modem.")

        return self.scenario.channel(self.reference_transmitter, self)

    @property
    def received_power(self) -> float:
        """Average signal power received by this modem for its reference peer.

        Assumes 1 Watt if no peer has been configured.

        Returns:
            float: The average power in Watts.
        """

        # Return unit power if no reference transmitter peer has been configured
        if self.reference_transmitter is None:
            return 1.0

        return self.reference_transmitter.power
