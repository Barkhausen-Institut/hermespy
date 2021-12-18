# -*- coding: utf-8 -*-
"""HermesPy Modem base class."""

from __future__ import annotations
from typing import List, Type, TYPE_CHECKING, Optional
from abc import abstractmethod
from ruamel.yaml import SafeRepresenter, MappingNode, ScalarNode
from scipy.constants import speed_of_light
import numpy as np
import numpy.random as rnd

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


from hermespy.precoding import SymbolPrecoding
from hermespy.coding import EncoderManager
from hermespy.modem.waveform_generator import WaveformGenerator
from hermespy.modem.rf_chain import RfChain
from hermespy.core.device import Transmitter, Receiver

if TYPE_CHECKING:

    from hermespy.scenario import Scenario
    from hermespy.channel import Channel


class Modem(Transmitter, Receiver):
    """HermesPy representation of a wireless communication modem.

    Modems may transmit or receive information in form of bit streams.

    In HermesPy, a modem is the basis of every simulation entity which may transmit or receive
    electromagnetic waveforms.

    The modem consists of an analog RF chain, a waveform generator, and can be used
    either for transmission or reception of a given technology.
    """

    yaml_tag = 'Modem'
    __scenario: Optional[Scenario]
    __position: Optional[np.ndarray]
    __orientation: Optional[np.ndarray]
    __topology: np.ndarray
    __carrier_frequency: float
    __linear_topology: bool
    __encoder_manager: EncoderManager
    __precoding: SymbolPrecoding
    __waveform_generator: Optional[WaveformGenerator]
    __rf_chain: RfChain
    __random_generator: Optional[rnd.Generator]

    def __init__(self,
                 carrier_frequency: Optional[float] = None,
                 encoding: Optional[EncoderManager] = None,
                 precoding: Optional[SymbolPrecoding] = None,
                 waveform: Optional[WaveformGenerator] = None,
                 rfchain: Optional[RfChain] = None,
                 random_generator: Optional[rnd.Generator] = None) -> None:
        """Object initialization.

        Args:
            scenario (Scenario):
                The scenario this modem is attached to.

            topology (np.ndarray, optional)
        """

        self.__carrier_frequency = 800e6
        self.__encoder_manager = EncoderManager()
        self.__precoding = SymbolPrecoding(modem=self)
        self.__waveform_generator = None
        self.__power = 1.0
        self.__rf_chain = RfChain()
        self.__random_generator = random_generator

        if carrier_frequency is not None:
            self.carrier_frequency = carrier_frequency

        # If num_antennas is configured initialize the modem as a Uniform Linear Array
        # with half wavelength element spacing
        if num_antennas is not None:

            if topology is not None:
                raise ValueError("The num_antennas and topology parameters are mutually exclusive")

            # For a carrier frequency of 0.0 we will initialize all antennas at the same position.
            half_wavelength = 0.0
            if self.__carrier_frequency > 0.0:
                half_wavelength = .5 * speed_of_light / self.__carrier_frequency

            self.topology = half_wavelength * np.outer(np.arange(num_antennas), np.array([1., 0., 0.]))

        elif topology is not None:
            self.topology = topology

        if encoding is not None:
            self.__encoder_manager = encoding

        if precoding is not None:
            self.__precoding = precoding

        if waveform is not None:
            self.waveform_generator = waveform

        if rfchain is not None:
            self.rf_chain = rfchain

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
            RfChain.yaml_tag: node.__rf_chain,
        }

        if node.waveform_generator is not None:
            serialization[node.waveform_generator.yaml_tag] = node.waveform_generator

        mapping: MappingNode = representer.represent_mapping(cls.yaml_tag, serialization)

        if node.__position is not None:

            sequence = representer.represent_list(node.__position.tolist())
            sequence.flow_style = True
            mapping.value.append((ScalarNode('tag:yaml.org,2002:str', 'position'), sequence))

        if node.__orientation is not None:

            sequence = representer.represent_list(node.__orientation.tolist())
            sequence.flow_style = True
            mapping.value.append((ScalarNode('tag:yaml.org,2002:str', 'orientation'), sequence))

        return mapping

    @property
    def scenario(self) -> Scenario:
        """Access the scenario this modem is attached to.

        Returns:
            Scenario:
                The referenced scenario.

        Raises:
            RuntimeError: If the modem is currently floating.
        """

        if self.__scenario is None:
            raise RuntimeError("Error trying to access the scenario of a floating modem")

        return self.__scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Attach the modem to a specific scenario.

        This can only be done once to a floating modem.

        Args:
            scenario (Scenario): The scenario this modem should be attached to.

        Raises:
            RuntimeError: If the modem is already attached to a scenario.
        """

        if self.__scenario is not None:
            raise RuntimeError("Error trying to modify the scenario of an already attached modem")

        self.__scenario = scenario

    @property
    def is_attached(self) -> bool:
        """Is the modem currently attached to a scenario?

        Returns:
            bool: Attachment state.
        """

        return self.__scenario is not None

    @property
    @abstractmethod
    def index(self) -> int:
        """The index of this modem in the scenario.

        Returns:
            int:
                The index.
        """

        pass

    @property
    @abstractmethod
    def paired_modems(self) -> List[Modem]:
        """The modems connected to this modem over an active channel.

        Returns:
            List[Modem]:
                A list of paired modems.
        """

        pass

    @property
    def random_generator(self) -> rnd.Generator:
        """Access the random number generator assigned to this modem.

        This property will return the scenarios random generator if no random generator has been specifically set.

        Returns:
            numpy.random.Generator: The random generator.

        Raises:
            RuntimeError: If trying to access the random generator of a floating modem.
        """

        if self.__random_generator is not None:
            return self.__random_generator

        if self.__scenario is None:
            raise RuntimeError("Trying to access the random generator of a floating modem")

        return self.__scenario.random_generator

    @random_generator.setter
    def random_generator(self, generator: Optional[rnd.Generator]) -> None:
        """Modify the configured random number generator assigned to this modem.

        Args:
            generator (Optional[numpy.random.generator]): The random generator. None if not specified.
        """

        self.__random_generator = generator

    @property
    def position(self) -> np.ndarray:
        """Access the modem's position.

        Returns:
            np.array:
                The modem position in xyz-coordinates.
        """

        return self.__position

    @position.setter
    def position(self, position: np.ndarray) -> None:
        """Update the modem's position.

        Args:
            position (np.array):
                The modem's new position.

        Raises:
            ValueError: If `position` is not a vector with at most three entries.
        """
        
        if position.ndim != 1:
            raise ValueError("Modem position must be a vector")

        if len(position) > 3:
            raise ValueError("Modem position may have at most three dimensions")

        # Make vector 3D if less dimensions have been supplied
        if len(position) < 3:
            position = np.append(position, np.zeros(3 - len(position), dtype=float))

        self.__position = position

    @property
    def orientation(self) -> np.ndarray:
        """Access the modem's orientation.

        Returns:
            np.array:
                The modem orientation as a normalized quaternion.
        """

        return self.__orientation

    @orientation.setter
    def orientation(self, orientation: np.ndarray) -> None:
        """Update the modem's orientation.

        Args:
            orientation(np.array):
                The new modem orientation.

        Raises:
            ValueError: If `orientation` is not a vector with 3 entries.
        """

        if orientation.ndim != 1 or len(orientation) != 3:
            raise ValueError("Modem orientation must be a three-dimensional vector")

        self.__orientation = orientation

    @property
    def topology(self) -> np.ndarray:
        """Access the configured sensor array topology.

        Returns:
            np.ndarray:
                A matrix of m x 3 entries describing the sensor array topology.
                Each row represents the xyz-location of a single antenna within an array of m antennas.
        """

        return self.__topology

    @topology.setter
    def topology(self, topology: np.ndarray) -> None:
        """Update the configured sensor array topology.

        Args:
            topology (np.ndarray):
                A matrix of m x 3 entries describing the sensor array topology.
                Each row represents the xyz-location of a single antenna within an array of m antennas.

        Raises:
            ValueError:
                If the first dimension `topology` is smaller than 1 or its second dimension is larger than 3.
        """

        if topology.ndim > 2:
            raise ValueError("The topology array must be of dimension 2")

        # If the topology is one-dimensional, we assume a ULA and therefore simply expand the second dimension
        if topology.ndim == 1:
            topology = topology.T[:, np.newaxis]

        if topology.shape[0] < 1:
            raise ValueError("The topology must contain at least one sensor")

        if topology.shape[1] > 3:
            raise ValueError("Second topology dimension may contain at most 3 fields (xyz)")

        # The second dimension may have less than 3 entries, in this case we expand by zeros
        if topology.shape[1] < 3:
            topology = np.append(topology, np.zeros((topology.shape[0], 3-topology.shape[1]), dtype=float), axis=1)

        self.__topology = topology

        # Automatically detect linearity in default configurations, where all sensor elements
        # are oriented along the local x-axis.
        axis_sums = np.sum(self.__topology, axis=0)
        self.__linear_topology = ((axis_sums[1] + axis_sums[2]) < 1e-10)

    @property
    def linear_topology(self) -> bool:
        """Access the configured linearity flag.

        Returns:
            bool:
                A boolean flag indicating whether this array is considered to be one-dimensional.
        """

        return self.__linear_topology

    @property
    def num_antennas(self) -> int:
        """The number of physical antennas available to the modem.

        For a transmitter this represents the number of transmitting antennas,
        or a receiver the respective receiving ones

        Returns:
            int:
                The number of physical antennas available to the modem.
        """

        return self.__topology.shape[0]

    @property
    def num_streams(self) -> int:
        """The number of data streams handled by the modem.

        The number of data streams is always less or equal to the number of available antennas `num_antennas`.

        Returns:
            int:
                The number of data streams generated by the modem.
        """

        # For now, stream compression will not be supported
        return self.num_antennas

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
        """Access the modem's configured waveform generator.

        Returns:
            WaveformGenerator:
                Handle to the modem's `WaveformGenerator` instance.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, waveform_generator: WaveformGenerator) -> None:
        """Configure the modem's waveform generator.

        This modifies the referenced modem within the `waveform_generator` to this modem!

        Args:
            waveform_generator (WaveformGenerator):
                The new waveform generator instance.
        """

        self.__waveform_generator = waveform_generator
        self.__waveform_generator.modem = self

    @property
    def rf_chain(self) -> RfChain:
        """Access the modem's configured RF chain.

        Returns:
            RfChain:
                Handle to the modem's `RfChain` instance.
        """

        return self.__rf_chain

    @rf_chain.setter
    def rf_chain(self, rf_chain: RfChain) -> None:
        """Configure the modem's RF chain

        Args:
            rf_chain (RfChain):
                The new waveform `RfChain` instance.
        """

        self.__rf_chain = rf_chain

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
    @abstractmethod
    def reference_channel(self) -> Channel:
        """Reference channel from the scenario channel matrix.

        By default the first channel within the matrix.

        Returns:
            Channel: The reference channel.
        """
        ...
