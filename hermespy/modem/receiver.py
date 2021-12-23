# -*- coding: utf-8 -*-
"""HermesPy Receiving Modem."""

from __future__ import annotations
from math import floor
from typing import TYPE_CHECKING, Type, List, Optional, Union, Tuple

import numpy as np
import numpy.random as rnd
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap

from hermespy.channel import Channel, ChannelStateDimension, ChannelStateInformation
from hermespy.modem import Modem
from hermespy.precoding import SymbolPrecoding
from hermespy.modem.waveform_generator import WaveformGenerator
from hermespy.noise import Noise
from hermespy.signal import Signal

if TYPE_CHECKING:
    from hermespy.scenario import Scenario
    from .transmitter import Transmitter

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.4"
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

    def receive(self, rf_signals: Union[List[Signal], Signal],
                noise_variance: float = 0.0) -> Signal:
        """Receive and down-mix radio-frequency-band signals to base-band signals over the receiver's hardware chain.

        Args:

            rf_signals (Union[List[Signal], Signal]):
                List containing radio-frequency band signal models impinging onto the receiver

            noise_variance (float, optional):
                Variance (i.e. power) of the thermal noise added to the signals during reception.

        Returns:

            Signal:
                A superposition of all `rf_signals` mixed to the base-band, distorted by noise and
                hardware-effects.

        Raises:

            ValueError:
                If the first dimension of each rf signal does not match the number of receive antennas.
                If `noise_variance` is smaller than zero.
        """

        baseband_signal = Signal.empty(self.waveform_generator.sampling_rate,
                                       carrier_frequency=self.carrier_frequency,
                                       num_streams=self.num_antennas)

        # Down-mix each rf-band signal to base-band and superimpose them at the receiver-side
        if isinstance(rf_signals, Signal):
            baseband_signal.superimpose(rf_signals)

        else:
            for rf_signal in rf_signals:
                baseband_signal.superimpose(rf_signal)

        # Scale resulting signal to unit power (relative to the configured transmitter reference)
        baseband_signal.samples /= np.sqrt(self.received_power)

        # Add receive noise
        noisy_signal = baseband_signal.copy()
        noisy_signal.samples = self.__noise.add_noise(baseband_signal.samples, noise_variance)

        # Simulate the radio-frequency chain
        received_signal = noisy_signal.copy()
        received_signal.samples = self.rf_chain.receive(received_signal.samples)

        # Return resulting base-band superposition
        return received_signal

    def demodulate(self, baseband_signal: Signal,
                   channel_state: ChannelStateInformation,
                   noise_variance: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            baseband_signal (Signal):
                Received signal noisy base-band signal.

            channel_state (ChannelStateInformation):
                State of the channel over which `baseband_signal` has been propagated.

            noise_variance (float, optional):
                Power of the incoming noise, required for equalization and channel estimation.

        Returns:
            (np.array, np.ndarray):
                - Detected bits as a list of data blocks for the drop.
                - Detected symbols after decoding.

        Raises:
            ValueError: If the first dimension of `input_signals` does not match the number of receive antennas.
        """

        if baseband_signal.num_streams != self.num_antennas:
            raise ValueError("Number of input signals must be equal to the number of antennas")

        if channel_state.num_receive_streams != self.num_antennas:
            raise ValueError("Number of channel state receive streams does not match the number of receiver antennas")

        num_samples = baseband_signal.num_samples

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
        for stream_idx, (rx_signal, stream_transform) in enumerate(zip(baseband_signal.samples,
                                                                       channel_state.received_streams())):

            frame = self.waveform_generator.synchronization.synchronize(rx_signal, stream_transform)
            frames.append(frame)

        frames = np.array(frames, dtype=object)

        # Demodulate the parallel frames arriving at each stream,
        # then decode the (inverse) precoding over all stream frames
        decoded_symbols = np.empty(0, dtype=complex)
        for frame_streams in frames.transpose(1, 0, 2):

            # Demodulate each frame separately
            symbols: List[np.ndarray] = []
            channel_states: List[ChannelStateInformation] = []
            noises: List[complex] = []

            for stream in frame_streams:

                # Demodulate the frame into data symbols
                s_symbols, s_channel_state, s_noise = self.waveform_generator.demodulate(*stream, noise_variance)

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

        # We're finally done, blow the fanfares, throw confetti, etc.
        return data_bits, decoded_symbols

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
