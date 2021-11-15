# -*- coding: utf-8 -*-
"""Channel model for wireless transmission links."""

from __future__ import annotations
from typing import Type, Tuple, TYPE_CHECKING, Optional
from abc import ABC
from itertools import product

import numpy as np
import numpy.random as rnd
from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode, MappingNode
from numba import jit, complex128
from sparse import DOK, COO

if TYPE_CHECKING:
    from hermespy.scenario import Scenario
    from hermespy.modem import Transmitter, Receiver

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class Channel(ABC):
    """Implements an ideal distortion-less channel.

    It also serves as a base class for all other channel models.

    For MIMO systems, the received signal is the addition of the signal transmitted at all
    antennas.
    The channel will provide `number_rx_antennas` outputs to a signal
    consisting of `number_tx_antennas` inputs. Depending on the channel model,
    a random number generator, given by `rnd` may be needed. The sampling rate is
    the same at both input and output of the channel, and is given by `sampling_rate`
    samples/second.

    Attributes:

        __active (bool):
            Flag enabling signal propagation over this specific channel.
            Enabled by default, may be disabled to easily debug scenarios.

        __transmitter (Optional[Transmitter]):
            Handle to the wireless modem transmitting into this channel.
            If set to `None`, this channel instance is considered floating.

        __receiver (Optional[Receiver]):
            Handle to the wireless modem receiving from this channel.
            If set to `None`, this channel instance is considered floating.

        __gain (float):
            Linear factor by which signals propagated over this channel will be scaled.
            1.0 by default, i.e. no free-space propagation losses are considered in the default channel.

        __random_generator (Optional[numpy.random.Generator]):
            Random generator object used to generate pseudo-random number sequences for this channel instance.
            If set to `None`, the channel will instead access the random generator of `scenario`.

        __scenario (Optional[Scenario]):
            HermesPy scenario description this channel belongs to.
            If set to `None`, this channel instance is considered floating.

        impulse_response_interpolation (bool):
            Allow for the impulse response to be resampled and interpolated.
    """

    yaml_tag = u'Channel'
    yaml_matrix = True
    __active: bool
    __transmitter: Optional[Transmitter]
    __receiver: Optional[Receiver]
    __gain: float
    __random_generator: Optional[rnd.Generator]
    __scenario: Optional[Scenario]
    impulse_response_interpolation: bool

    def __init__(self,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 scenario: Optional[Scenario] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 random_generator: Optional[rnd.Generator] = None,
                 impulse_response_interpolation: bool = True
                 ) -> None:
        """Class constructor.

        Args:
            transmitter (Transmitter, optional):
                The modem transmitting into this channel.

            receiver (Receiver, optional):
                The modem receiving from this channel.

            scenario (Scenario, optional):
                Scenario this channel is attached to.

            active (bool, optional):
                Channel activity flag.
                Activated by default.

            gain (float, optional):
                Channel power gain.
                1.0 by default.
                
            random_generator (rnd.Generator, optional):
                Generator object for random number sequences.

            impulse_response_interpolation (bool, optional):
                Allow the impulse response to be interpolated during sampling.
        """

        # Default parameters
        self.__active = True
        self.__transmitter = None
        self.__receiver = None
        self.__gain = 1.0
        self.__scenario = None
        self.recent_response = None
        self.impulse_response_interpolation = impulse_response_interpolation

        self.random_generator = random_generator
        self.scenario = scenario

        if transmitter is not None:
            self.transmitter = transmitter

        if receiver is not None:
            self.receiver = receiver

        if active is not None:
            self.active = active

        if gain is not None:
            self.gain = gain

    @property
    def active(self) -> bool:
        """Access channel activity flag.

        Returns:
            bool:
                Is the channel currently activated?
        """

        return self.__active

    @active.setter
    def active(self, active: bool) -> None:
        """Modify the channel activity flag.

        Args:
            active (bool):
                Is the channel currently activated?
        """

        self.__active = active

    @property
    def transmitter(self) -> Transmitter:
        """Access the modem transmitting into this channel.

        Returns:
            Transmitter: A handle to the modem transmitting into this channel.
        """

        return self.__transmitter

    @transmitter.setter
    def transmitter(self, new_transmitter: Transmitter) -> None:
        """Configure the modem transmitting into this channel.

        Args:
            new_transmitter (Transmitter): The transmitter to be configured.

        Raises:
            RuntimeError: If a transmitter is already configured.
        """

        if self.__transmitter is not None:
            raise RuntimeError("Overwriting a transmitter configuration is not supported")

        self.__transmitter = new_transmitter

    @property
    def receiver(self) -> Receiver:
        """Access the modem receiving from this channel.

        Returns:
            Receiver: A handle to the modem receiving from this channel.
        """

        return self.__receiver

    @receiver.setter
    def receiver(self, new_receiver: Receiver) -> None:
        """Configure the modem receiving from this channel.

        Args:
            new_receiver (Receiver): The receiver to be configured.

        Raises:
            RuntimeError: If a receiver is already configured.
        """

        if self.__receiver is not None:
            raise RuntimeError("Overwriting a receiver configuration is not supported")

        self.__receiver = new_receiver

    @property
    def gain(self) -> float:
        """Access the channel gain.

        The default channel gain is 1.
        Realistic physical channels should have a gain less than one.

        Returns:
            float:
                The channel gain.
        """

        return self.__gain

    @gain.setter
    def gain(self, value: float) -> None:
        """Modify the channel gain.

        Args:
            value (float):
                The new channel gain.

        Raises:
            ValueError: If gain is smaller than zero.
        """

        if value < 0.0:
            raise ValueError("Channel gain must be greater or equal to zero")

        self.__gain = value
        
    @property
    def random_generator(self) -> rnd.Generator:
        """Access the random number generator assigned to this channel.

        This property will return the scenarios random generator if no random generator has been specifically set.

        Returns:
            numpy.random.Generator: The random generator.

        Raises:
            RuntimeError: If trying to access the random generator of a floating channel.
        """

        if self.__scenario is None:
            raise RuntimeError("Trying to access the random generator of a floating channel")

        if self.__random_generator is None:
            return self.__scenario.random_generator

        return self.__random_generator

    @random_generator.setter
    def random_generator(self, generator: Optional[rnd.Generator]) -> None:
        """Modify the configured random number generator assigned to this channel.

        Args:
            generator (Optional[numpy.random.generator]): The random generator. None if not specified.
        """

        self.__random_generator = generator
        
    @property
    def scenario(self) -> Scenario:
        """Access the scenario this channel is attached to.

        Returns:
            Scenario:
                The referenced scenario.

        Raises:
            RuntimeError: If the channel is currently floating.
        """

        if self.__scenario is None:
            raise RuntimeError("Error trying to access the scenario of a floating channel")

        return self.__scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Attach the channel to a specific scenario.

        This can only be done once to a floating channel.

        Args:
            scenario (Scenario): The scenario this channel should be attached to.

        Raises:
            RuntimeError: If the channel is already attached to a scenario.
        """

        if self.__scenario is not None:
            raise RuntimeError("Error trying to modify the scenario of an already attached channel")

        self.__scenario = scenario

    @property
    def num_inputs(self) -> int:
        """The number of streams feeding into this channel.

        Actually shadows the `num_streams` property of his channel's transmitter.

        Returns:
            int:
                The number of input streams.

        Raises:
            RuntimeError: If the channel is currently floating.
        """

        if self.__transmitter is None:
            raise RuntimeError("Error trying to access the number of inputs property of a floating channel")

        return self.__transmitter.num_antennas

    @property
    def num_outputs(self) -> int:
        """The number of streams emerging from this channel.

        Actually shadows the `num_streams` property of his channel's receiver.

        Returns:
            int:
                The number of output streams.

        Raises:
            RuntimeError: If the channel is currently floating.
        """

        if self.__receiver is None:
            raise RuntimeError("Error trying to access the number of outputs property of a floating channel")

        return self.__receiver.num_antennas

    @property
    def indices(self) -> Tuple[int, int]:
        """The indices of this channel within the scenarios channel matrix.

        Returns:
            int:
                Transmitter index.
            int:
                Receiver index.
        """

        return self.__transmitter.index, self.__receiver.index

    def propagate(self, transmitted_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Modifies the input signal and returns it after channel propagation.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of one's.
        The routine samples a new impulse response.

        Args:

            transmitted_signal (np.ndarray):
                Input signal antenna signals to be propagated of this channel instance.
                The array is expected to be two-dimensional with shape `num_transmit_antennas`x`num_samples`.

        Returns:
            (np.ndarray, np.ndarray):
                Tuple of the distorted signal after propagation and the channel impulse response.

                The propagated signal is a two-dimensional array with shape
                `num_receive_antennas`x`num_propagated_samples`.
                Note that the channel may append samples to the propagated signal,
                so that `num_propagated_samples` is generally not equal to `num_samples`.

                The impulse response is a 4-dimensional array of size
                `T x num_receive_antennas x num_transmit_antennas x (L+1)`,
                where `L` is the maximum path delay (in samples).
        Raises:

            ValueError:
                If the first dimension of `transmitted_signal` is not one or the number of transmitting antennas.

            RuntimeError:
                If the scenario configuration is not supported by the default channel model.

            RuntimeError:
                If the channel is currently floating.
        """

        if transmitted_signal.ndim != 2:
            raise ValueError("Transmitted signal must be a matrix (an array of two dimensions)")

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making propagation simulation impossible")

        if transmitted_signal.shape[0] != self.transmitter.num_antennas:
            raise ValueError("Number of transmitted signal streams does not match number of transmit antennas")

        # If the channel is inactive, propagation will result in signal loss
        # This is modeled by returning an zero-length signal and impulse-response (in time-domain) after propagation
        if not self.active:
            return (np.empty((self.receiver.num_antennas, 0), dtype=complex),
                    np.empty((0, self.transmitter.num_antennas, self.receiver.num_antennas, 1), dtype=complex))

        # Generate the channel's impulse response
        num_signal_samples = transmitted_signal.shape[1]
        impulse_response = self.impulse_response(np.arange(num_signal_samples) / self.scenario.sampling_rate)

        # The maximum delay (in samples) is modeled by the last impulse response dimension
        num_delay_samples = impulse_response.shape[3] - 1

        # Propagate the signal
        received_signal = np.zeros((self.receiver.num_antennas, transmitted_signal.shape[1] + num_delay_samples),
                                   dtype=complex)

        for delay_index in range(impulse_response.shape[3]):
            for tx_idx, rx_idx in product(range(self.transmitter.num_antennas), range(self.receiver.num_antennas)):

                delayed_signal = impulse_response[:, rx_idx, tx_idx, delay_index] * transmitted_signal[tx_idx, :]
                received_signal[rx_idx, delay_index:delay_index+num_signal_samples] += delayed_signal

        return received_signal, impulse_response

    def impulse_response(self, timestamps: np.ndarray) -> np.ndarray:
        """Calculate the channel impulse responses.

        This method can be used for instance by the transceivers to obtain the channel state
        information.

        Args:
            timestamps (np.ndarray):
                Time instants with length `T` to calculate the response for.

        Returns:
            np.ndarray:
                Impulse response in all `number_rx_antennas` x `number_tx_antennas`.
                4-dimensional array of size `T x number_rx_antennas x number_tx_antennas x (L+1)`
                where `L` is the maximum path delay (in samples). For the ideal
                channel in the base class, `L = 0`.
        """

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making impulse response simulation impossible")

        # MISO case
        if self.receiver.num_antennas == 1:
            impulse_responses = np.tile(np.ones((1, self.transmitter.num_antennas), dtype=complex),
                                        (timestamps.size, 1, 1))

        # SIMO case
        elif self.transmitter.num_antennas == 1:
            impulse_responses = np.tile(np.ones((self.receiver.num_antennas, 1), dtype=complex),
                                        (timestamps.size, 1, 1))

        # MIMO case
        else:
            impulse_responses = np.tile(np.eye(self.receiver.num_antennas, self.transmitter.num_antennas, dtype=complex),
                                        (timestamps.size, 1, 1))

        # Scale by channel gain and add dimension for delay response
        impulse_responses = self.gain * np.expand_dims(impulse_responses, axis=3)

        # Return resulting impulse response
        return impulse_responses

    @staticmethod
    @jit(complex128[:, :](complex128[:, :]), nopython=True)
    def delay_matrix(power_delay_profile: np.ndarray) -> np.ndarray:
        """Transform a channel impulse response power delay profile to a linear transformation matrix.

        Args:
            power_delay_profile (np.ndarray):
                A matrix of dimension TxL+1 where T is the sampled time and L is the sampled delay, respectively.

        Returns:
            np.ndarray:
                A complex (T+L)xT transformation matrix representing the `power_delay_profile`.
        """

        num_timestamps = power_delay_profile.shape[0]
        num_taps = power_delay_profile.shape[1]
        convolution = np.zeros((num_timestamps + num_taps - 1, num_timestamps), dtype=complex128)

        time_indices = np.arange(num_timestamps)
        for delay_index in range(power_delay_profile.shape[1]):
            for time_index in time_indices:

                convolution[delay_index + time_index, time_index] = power_delay_profile[time_index, delay_index]

        return convolution

    @staticmethod
    def power_delay_profile(delay_matrix: np.ndarray, num_delay_taps: int, num_timestamps: int = 0) -> np.ndarray:
        """Transform a linear transformation matrix to a power delay profile.

        Args:
            delay_matrix (np.ndarray):
                A complex TxT transformation matrix representing a `power_delay_profile`.

            num_delay_taps (int):
                The number of L delay taps.
        Returns:
            np.ndarray:
                A matrix of dimension TxL+1 where T is the sampled time and L is the sampled delay, respectively.
        """

        if num_timestamps == 0:
            num_timestamps = delay_matrix.shape[0] - num_delay_taps + 1

        power_delay_profile = np.empty((num_timestamps, num_delay_taps), dtype=complex)

        # The delay profile is contained within the off-diagonal elements of the delay matrix
        for delay_idx in range(num_delay_taps):

            diagonal_elements = np.diag(delay_matrix, -delay_idx)
            power_delay_profile[0:len(diagonal_elements), delay_idx] = diagonal_elements

        return power_delay_profile

    @staticmethod
#    @jit(complex128[:, :, :, :](complex128[:, :, :, :]), nopython=True)
    def impulse_transformation(impulse_response: np.ndarray) -> COO:
        """Convert a channel impulse response to a linear transformation tensor.

        Args:
            impulse_response (np.ndarray):
                T x N_Rx x N_Tx x L+1 Channel impulse response tensor.
                See `impulse_response` for further details.

        Returns:
            DOK:
                Sparse linear transformation tensor of dimension N_Rx x N_Tx x T+L x T.
                Note that the slice over the first snd last dimension will usually form a lower triangular matrix.
        """

        num_rx = impulse_response.shape[1]
        num_tx = impulse_response.shape[2]
        num_taps = impulse_response.shape[3]
        num_out = impulse_response.shape[0] + num_taps - 1
        num_in = impulse_response.shape[0]

        in_ids = np.repeat(np.arange(num_in), num_taps)
        out_ids = np.array([np.arange(num_taps) + t for t in range(num_in)]).flatten()
        rx_ids = np.arange(num_rx)
        tx_ids = np.arange(num_tx)

        coordinates = [rx_ids.repeat(num_tx * num_taps * num_in),
                       tx_ids.repeat(num_rx * num_taps * num_in).reshape((num_tx, -1), order='F').flatten(),
                       np.tile(out_ids, num_rx * num_tx),
                       np.tile(in_ids, num_rx * num_tx)]
        data = impulse_response.transpose((1, 2, 0, 3)).flatten()

        transformation = COO(coordinates, data, shape=(num_rx, num_tx, num_out, num_in))
        return transformation

    @property
    def min_sampling_rate(self) -> float:
        """Minimal sampling rate required to adequately model the channel.

        Returns:
            float: The minimal sampling rate in Hz.
        """

        # Since the default channel is time-invariant, there are no sampling rate requirements
        return 0.0

    @classmethod
    def to_yaml(cls: Type[Channel], representer: SafeRepresenter, node: Channel) -> MappingNode:
        """Serialize a channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Channel):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'active': node.__active,
            'gain': node.__gain
        }

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml

    @classmethod
    def from_yaml(cls: Type[Channel], constructor: SafeConstructor,  node: MappingNode) -> Channel:
        """Recall a new `Channel` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Channel` serialization.

        Returns:
            Channel:
                Newly created `Channel` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.

        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)

        seed = state.pop('seed', None)
        if seed is not None:
            state['random_generator'] = rnd.default_rng(seed)

        return cls(**state)
