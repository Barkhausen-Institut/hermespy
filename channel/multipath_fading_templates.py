# -*- coding: utf-8 -*-
"""Multipath Fading Channel prebuilt templates."""

from __future__ import annotations
import numpy as np
from enum import Enum
from typing import TYPE_CHECKING, Optional, Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, ScalarNode

from channel import MultipathFadingChannel

if TYPE_CHECKING:
    from modem import Transmitter, Receiver

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class MultipathFadingCost256(MultipathFadingChannel):
    """COST256 Multipath Fading Channel models."""

    class TYPE(Enum):
        """Supported model types."""

        URBAN = 0
        RURAL = 1
        HILLY = 2

    yaml_tag = u'COST256'
    yaml_matrix = True
    __model_type: TYPE

    def __init__(self,
                 model_type: TYPE = 0,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 num_sinusoids: Optional[float] = None,
                 los_angle: Optional[float] = None,
                 doppler_frequency: Optional[float] = None,
                 los_doppler_frequency: Optional[float] = None,
                 transmit_precoding: Optional[np.ndarray] = None,
                 receive_postcoding: Optional[np.ndarray] = None) -> None:
        """Model initialization.

        Args:
            model_type (TYPE): The model type..
            transmitter (Transmitter, optional): The modem transmitting into this channel.
            receiver (Receiver, optional): The modem receiving from this channel.
            active (bool, optional): Channel activity flag.
            gain (float, optional): Channel power gain.
            num_sinusoids (int, optional): Number of sinusoids used to sample the statistical distribution.
            los_angle (float, optional): Angle phase of the line of sight component within the statistical distribution.
            doppler_frequency (float, optional): Doppler frequency shift of the statistical distribution.
            transmit_precoding (np.ndarray): Transmit precoding matrix.
            receive_postcoding (np.ndarray): Receive postcoding matrix.

        Raises:
           ValueError:
                If `model_type` is not supported.
                If `los_angle` is defined in HILLY model type.
        """

        if model_type == self.TYPE.URBAN:

            delays = 1e-6 * np.asarray([0, .217, .512, .514, .517, .674, .882, 1.230, 1.287, 1.311, 1.349,
                                        1.533, 1.535, 1.622, 1.818, 1.836, 1.884, 1.943, 2.048, 2.140])
            power_db = np.asarray([-5.7, - 7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9, -17.1,
                                   -17.4, -19.0, -19.0, -19.8, -21.5, -21.6, -22.1, -22.6, -23.5, -24.3])
            rice_factors = np.zeros(delays.shape)

        elif model_type == self.TYPE.RURAL:

            delays = 1e-6 * np.asarray([0, .042, .101, .129, .149, .245, .312, .410, .469, .528])
            power_db = np.asarray([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4])
            rice_factors = np.zeros(delays.shape)

        elif model_type == self.TYPE.HILLY:

            if los_angle is not None:
                raise ValueError("Model type HILLY does not support line of sight angle configuration")

            delays = 1e-6 * np.asarray([0, .356, .441, .528, .546, .609, .625, .842, .916, .941, 15.0,
                                        16.172, 16.492, 16.876, 16.882, 16.978, 17.615, 17.827, 17.849, 18.016])
            power_db = np.asarray([-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3, -17.7,
                                   -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9, -30.0, -30.7])
            rice_factors = np.hstack((np.inf, np.zeros(delays.size - 1)))
            los_angle = np.arccos(.7)

        else:
            raise ValueError("Requested model type not supported")

        self.__model_type = model_type

        # Convert power and normalize
        power_profile = 10 ** (power_db / 10)
        power_profile /= sum(power_profile)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(self,
                                        delays,
                                        power_profile,
                                        rice_factors,
                                        transmitter,
                                        receiver,
                                        active,
                                        gain,
                                        num_sinusoids,
                                        los_angle,
                                        doppler_frequency,
                                        los_doppler_frequency,
                                        transmit_precoding,
                                        receive_postcoding)

    @property
    def model_type(self) -> TYPE:
        """Access the configured model type.

        Returns:
            MultipathFadingCost256.TYPE: The configured model type.
        """

        return self.__model_type

    @classmethod
    def to_yaml(cls: Type[MultipathFadingCost256], representer: SafeRepresenter,
                node: MultipathFadingCost256) -> MappingNode:
        """Serialize a channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MultipathFadingCost256):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'type': node.model_type.name,
            'active': node.active,
            'gain': node.gain,
            'num_sinusoids': node.num_sinusoids,
            'los_angle': node.los_angle,
            'doppler_frequency': node.doppler_frequency,
            'los_doppler_frequency': node.los_doppler_frequency,
            'transmit_precoding': node.transmit_precoding,
            'receive_postcoding': node.receive_postcoding,
        }

        if node.model_type is MultipathFadingCost256.TYPE.HILLY:
            state.pop('los_angle')

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml

    @classmethod
    def from_yaml(cls: Type[MultipathFadingCost256], constructor: SafeConstructor, node: MappingNode) -> \
            MultipathFadingCost256:
        """Recall a new `MultipathFadingCost256` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MultipathFadingCost256` serialization.

        Returns:
            Channel:
                Newly created `MultipathFadingCost256` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.
        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            raise RuntimeError("Cost256 channel configurations require at least a model specification")

        state = constructor.construct_mapping(node)

        model_type = state.pop('model_type', None)
        if model_type is None:
            raise RuntimeError("Cost256 channel configurations require at least a model specification")

        state['model_type'] = cls.TYPE[model_type]
        return cls(**state)


class MultipathFading5GTDL(MultipathFadingChannel):
    """5G TDL Multipath Fading Channel models."""

    class TYPE(Enum):
        """Supported model types."""

        A = 0
        B = 1
        C = 2
        D = 4
        E = 5

    yaml_tag = u'5GTDL'
    yaml_matrix = True
    model_type: TYPE
    __rms_delay: float

    def __init__(self,
                 model_type: TYPE = 0,
                 rms_delay: float = 0.0,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 num_sinusoids: Optional[float] = None,
                 los_angle: Optional[float] = None,
                 doppler_frequency: Optional[float] = None,
                 los_doppler_frequency: Optional[float] = None,
                 transmit_precoding: Optional[np.ndarray] = None,
                 receive_postcoding: Optional[np.ndarray] = None) -> None:
        """Model initialization.

        Args:
            model_type (TYPE): The model type.
            rms_delay (float): Root-Mean-Squared delay.
            transmitter (Transmitter, optional): The modem transmitting into this channel.
            receiver (Receiver, optional): The modem receiving from this channel.
            active (bool, optional): Channel activity flag.
            gain (float, optional): Channel power gain.
            num_sinusoids (int, optional): Number of sinusoids used to sample the statistical distribution.
            los_angle (float, optional): Angle phase of the line of sight component within the statistical distribution.
            doppler_frequency (float, optional): Doppler frequency shift of the statistical distribution.
            transmit_precoding (np.ndarray): Transmit precoding matrix.
            receive_postcoding (np.ndarray): Receive postcoding matrix.

        Raises:
            ValueError:
                If `model_type` is not supported.
                If `rms_delay` is smaller than zero.
                If `los_angle` is specified in combination with `model_type` D or E.
        """

        if rms_delay < 0.0:
            raise ValueError("Root-Mean-Squared delay must be greater or equal to zero")

        self.__rms_delay = rms_delay

        if model_type == self.TYPE.A:

            normalized_delays = np.asarray([0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618,
                                            1.5375, 1.8978, 2.2242, 2.1717, 2.4942, 2.5119, 3.0582,
                                            4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
            power_db = np.asarray([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5,
                                   -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8,
                                   -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7])
            rice_factors = np.zeros(normalized_delays.shape)

        elif model_type == self.TYPE.B:

            normalized_delays = np.asarray([0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697,
                                            0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187,
                                            4.1067, 4.2790, 4.7834])
            power_db = np.asarray([0, -2.2, -4, -3.2, -9.8, -3.2, -3.4, -5.2, -7.6, -3, -8.9, -9, -4.8,
                                   -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4, -14.9, -9.2, -11.3])
            rice_factors = np.zeros(normalized_delays.shape)

        elif model_type == self.TYPE.C:

            normalized_delays = np.asarray([0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935,
                                            0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003, 5.4902,
                                            5.6077, 6.3065, 6.6374, 7.0427, 8.6523])
            power_db = np.asarray([-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1,
                                   -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16, -15.7, -21.6,
                                   -22.8])
            rice_factors = np.zeros(normalized_delays.shape)

        elif model_type == self.TYPE.D:

            if los_doppler_frequency is not None:
                raise ValueError("Model type D does not support line of sight doppler frequency configuration")

            normalized_delays = np.asarray([0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424,
                                            9.708, 12.525])
            power_db = np.asarray([-13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8,
                                   -23.6, -24.8, -30.0, -27.7])
            rice_factors = np.zeros(normalized_delays.shape)
            rice_factors[0] = 13.3
            los_doppler_frequency = 0.7

        elif model_type == self.TYPE.E:

            if los_doppler_frequency is not None:
                raise ValueError("Model type E does not support line of sight doppler frequency configuration")

            normalized_delays = np.asarray([0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589,
                                            2.6426, 3.7136, 5.4524, 12.0034, 20.6519])
            power_db = np.asarray([-22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6,
                                   -22.3, -25.6, -20.2, -29.8, -29.2])
            rice_factors = np.zeros(normalized_delays.shape)
            rice_factors[0] = 22
            los_doppler_frequency = 0.7

        else:
            raise ValueError("Requested model type not supported")

        self.__model_type = model_type

        # Convert power and normalize
        power_profile = 10 ** (power_db / 10)
        power_profile /= sum(power_profile)

        # Scale delays
        delays = rms_delay * normalized_delays

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(self,
                                        delays,
                                        power_profile,
                                        rice_factors,
                                        transmitter,
                                        receiver,
                                        active,
                                        gain,
                                        num_sinusoids,
                                        los_angle,
                                        doppler_frequency,
                                        los_doppler_frequency,
                                        transmit_precoding,
                                        receive_postcoding)

    @property
    def model_type(self) -> TYPE:
        """Access the configured model type.

        Returns:
            MultipathFading5gTDL.TYPE: The configured model type.
        """

        return self.__model_type

    @classmethod
    def to_yaml(cls: Type[MultipathFading5GTDL], representer: SafeRepresenter,
                node: MultipathFading5GTDL) -> MappingNode:
        """Serialize a channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MultipathFading5GTDL):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'type': node.model_type.name,
            'rms_delay': node.__rms_delay,
            'active': node.active,
            'gain': node.gain,
            'num_sinusoids': node.num_sinusoids,
            'los_angle': node.los_angle,
            'doppler_frequency': node.doppler_frequency,
            'los_doppler_frequency': node.los_doppler_frequency,
            'transmit_precoding': node.transmit_precoding,
            'receive_postcoding': node.receive_postcoding,
        }

        if node.model_type is MultipathFading5GTDL.TYPE.C or MultipathFading5GTDL.TYPE.E:
            state.pop('los_doppler_frequency')

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml

    @classmethod
    def from_yaml(cls: Type[MultipathFading5GTDL], constructor: SafeConstructor, node: MappingNode) -> \
            MultipathFading5GTDL:
        """Recall a new `MultipathFading5GTDL` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MultipathFading5GTDL` serialization.

        Returns:
            Channel:
                Newly created `MultipathFading5GTDL` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.
        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            raise RuntimeError("5G TDL channel configurations require at least a model specification")

        state = constructor.construct_mapping(node)

        model_type = state.pop('type', None)
        if model_type is None:
            raise RuntimeError("5G TDL channel configurations require at least a model specification")

        state['model_type'] = cls.TYPE[model_type]
        return cls(**state)


class MultipathFadingExponential(MultipathFadingChannel):
    """Exponential Multipath Fading Channel models."""

    yaml_tag = u'Exponential'
    yaml_matrix = True
    __exponential_truncation: float = 1e-5
    __tap_interval: float
    __rms_delay: float

    def __init__(self,
                 tap_interval: float = 0.0,
                 rms_delay: float = 0.0,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 num_sinusoids: Optional[float] = None,
                 los_angle: Optional[float] = None,
                 doppler_frequency: Optional[float] = None,
                 los_doppler_frequency: Optional[float] = None,
                 transmit_precoding: Optional[np.ndarray] = None,
                 receive_postcoding: Optional[np.ndarray] = None) -> None:
        """Exponential Multipath Channel Model initialization.

        Args:
            tap_interval (float, optional): Tap interval in seconds.
            rms_delay (float, optional): Root-Mean-Squared delay in seconds.
            transmitter (Transmitter, optional): The modem transmitting into this channel.
            receiver (Receiver, optional): The modem receiving from this channel.
            active (bool, optional): Channel activity flag.
            gain (float, optional): Channel power gain.
            num_sinusoids (int, optional): Number of sinusoids used to sample the statistical distribution.
            los_angle (float, optional): Angle phase of the line of sight component within the statistical distribution.
            doppler_frequency (float, optional): Doppler frequency shift of the statistical distribution.
            transmit_precoding (np.ndarray): Transmit precoding matrix.
            receive_postcoding (np.ndarray): Receive postcoding matrix.

        Raises:
            ValueError: On invalid arguments.
        """

        if tap_interval <= 0.0:
            raise ValueError("Tap interval must be greater than zero")

        if rms_delay <= 0.0:
            raise ValueError("Root-Mean-Squared delay must be greater than zero")

        self.__tap_interval = tap_interval
        self.__rms_delay = rms_delay

        rms_norm = rms_delay / tap_interval

        # Calculate the decay exponent alpha based on an infinite power delay profile, in which case
        # rms_delay = exp(-alpha/2)/(1-exp(-alpha)), cf. geometric distribution.
        # Truncate the distributions for paths whose average power is very
        # small (less than exponential_truncation).

        alpha = -2 * np.log((-1 + np.sqrt(1 + 4 * rms_norm ** 2)) / (2 * rms_norm))
        max_delay_in_samples = int(np.ceil(np.log(MultipathFadingExponential.__exponential_truncation) / alpha))

        delays = np.arange(max_delay_in_samples + 1) * tap_interval
        power_profile = np.exp(-alpha * np.arange(max_delay_in_samples + 1))
        rice_factors = np.zeros(delays.shape)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(self,
                                        delays,
                                        power_profile,
                                        rice_factors,
                                        transmitter,
                                        receiver,
                                        active,
                                        gain,
                                        num_sinusoids,
                                        los_angle,
                                        doppler_frequency,
                                        los_doppler_frequency,
                                        transmit_precoding,
                                        receive_postcoding)

    @classmethod
    def to_yaml(cls: Type[MultipathFadingExponential], representer: SafeRepresenter,
                node: MultipathFadingExponential) -> MappingNode:
        """Serialize a channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MultipathFadingExponential):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'tap_interval': node.__tap_interval,
            'rms_delay': node.__rms_delay,
            'active': node.active,
            'gain': node.gain,
            'num_sinusoids': node.num_sinusoids,
            'los_angle': node.los_angle,
            'doppler_frequency': node.doppler_frequency,
            'los_doppler_frequency': node.los_doppler_frequency,
            'transmit_precoding': node.transmit_precoding,
            'receive_postcoding': node.receive_postcoding,
        }

        transmitter_index, receiver_index = node.indices
        return representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)

    @classmethod
    def from_yaml(cls: Type[MultipathFadingExponential], constructor: SafeConstructor, node: MappingNode) -> \
            MultipathFadingExponential:
        """Recall a new `MultipathFadingExponential` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MultipathFadingExponential` serialization.

        Returns:
            Channel:
                Newly created `MultipathFadingExponential` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.
        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)
        return cls(**state)
