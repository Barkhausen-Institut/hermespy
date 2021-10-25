from __future__ import annotations
from typing import Tuple, List, Type, TYPE_CHECKING, Optional
from abc import abstractmethod
from enum import Enum
import numpy as np
from ruamel.yaml import SafeRepresenter, MappingNode, ScalarNode

from modem.precoding import Precoding
from coding import EncoderManager
from modem.waveform_generator import WaveformGenerator
from modem.rf_chain import RfChain
from source.bits_source import BitsSource

if TYPE_CHECKING:
    from scenario import Scenario


class TransmissionMode(Enum):
    """Direction of transmission.
    """

    Rx = 1  # Receive mode
    Tx = 2  # Transmit mode


class Modem:
    """Implements a modem.

    The modem consists of an analog RF chain, a waveform generator, and can be used
    either for transmission or reception of a given technology.
    """

    yaml_tag = 'Modem'
    __scenario: Optional[Scenario]
    __position: Optional[np.ndarray]
    __orientation: Optional[np.ndarray]
    __topology: np.ndarray
    __carrier_frequency: float
    __sampling_rate: float
    __linear_topology: bool
    __encoder_manager: EncoderManager
    __precoding: Precoding
    __bits_source: BitsSource
    __waveform_generator: Optional[WaveformGenerator]
    __tx_power: float
    __rf_chain: RfChain

    def __init__(self,
                 scenario: Scenario = None,
                 position: np.array = None,
                 orientation: np.array = None,
                 topology: np.ndarray = None,
                 carrier_frequency: float = None,
                 sampling_rate: float = None,
                 bits: BitsSource = None,
                 encoding: EncoderManager = None,
                 precoding: Precoding = None,
                 waveform: WaveformGenerator = None,
                 tx_power: float = None,
                 rfchain: RfChain = None) -> None:
        """Object initialization.

        Args:
            scenario (Scenario):
                The scenario this modem is attached to.

            topology (np.ndarray, optional)
        """

        self.__scenario = None
        self.__position = None
        self.__orientation = None
        self.__topology = np.zeros((1, 3), dtype=np.float64)
        self.__carrier_frequency = 800e6
        self.__sampling_rate = 1e3
        self.__linear_topology = False
        self.__bits_source = BitsSource()
        self.__encoder_manager = EncoderManager()
        self.__precoding = Precoding()
        self.__waveform_generator = None
        self.__tx_power = 1.0
        self.__rf_chain = RfChain()

        if scenario is not None:
            self.scenario = scenario

        if position is not None:
            self.position = position

        if orientation is not None:
            self.orientation = orientation

        if topology is not None:
            self.topology = topology

        if carrier_frequency is not None:
            self.carrier_frequency = carrier_frequency

        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

        if bits is not None:
            self.bits_source = bits

        if encoding is not None:
            self.__encoder_manager = encoding

        if precoding is not None:
            self.__precoding = precoding

        if waveform is not None:
            self.waveform_generator = waveform

        if tx_power is not None:
            self.tx_power = tx_power

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
            "sampling_rate": node.__sampling_rate,
            "tx_power": node.__tx_power,
            BitsSource.yaml_tag: node.__bits_source,
            EncoderManager.yaml_tag: node.__encoder_manager,
            Precoding.yaml_tag: node.__precoding,
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
        number_of_samples = int(np.ceil(drop_duration * self.sampling_rate))
        timestamp = 0
        frame_index = 1

        num_code_bits = self.waveform_generator.bits_per_frame
        num_data_bits = self.encoder_manager.required_num_data_bits(num_code_bits)

        # Generate source data bits if none are provided
        if data_bits is None:
            data_bits = self.__bits_source.get_bits(num_data_bits)[0]

        # Make sure enough data bits were provided
        elif len(data_bits) < num_data_bits:
            raise ValueError("Number of provided data bits is insufficient to generate a single frame")

        # Apply channel coding to the source bits
        code_bits = self.encoder_manager.encode(data_bits, num_code_bits)

        while timestamp < number_of_samples:

            # Generate base-band waveforms
            frame, timestamp, initial_sample_num = self.waveform_generator.create_frame(
                timestamp, code_bits)

            if frame_index == 1:
                tx_signal, samples_delay = self._allocate_drop_size(
                    initial_sample_num, number_of_samples)

            tx_signal, samples_delay = self._add_frame_to_drop(
                initial_sample_num, samples_delay, tx_signal, frame)
            frame_index += 1

        tx_signal = self.rf_chain.send(tx_signal)
        tx_signal = self._adjust_tx_power(tx_signal)
        return tx_signal

    def _add_frame_to_drop(self, initial_sample_num: int,
                           samples_delay: int, tx_signal: np.ndarray,
                           frame: np.ndarray) -> Tuple[np.ndarray, int]:
        initial_sample_idx = samples_delay + initial_sample_num
        end_sample_idx = initial_sample_idx + frame.shape[1]

        if end_sample_idx > tx_signal.shape[1]:
            # last frame may be larger than allocated space, because of
            # filtering
            tx_signal = np.append(
                tx_signal, np.zeros((self.num_antennas, end_sample_idx - tx_signal.shape[1])), axis=1)

        tx_signal[:, initial_sample_idx:end_sample_idx] += frame
        return tx_signal, samples_delay

    def _allocate_drop_size(self, initial_sample_num: int,
                            number_of_samples: int) -> Tuple[np.ndarray, int]:
        if initial_sample_num < 0:
            # first frame may start before 0 because of filtering
            samples_delay = -initial_sample_num
        else:
            samples_delay = 0

        tx_signal = np.zeros((self.num_antennas, number_of_samples - initial_sample_num),
                             dtype=complex)
        return tx_signal, samples_delay

    def _adjust_tx_power(self, tx_signal: np.ndarray) -> np.ndarray:
        """Adjusts power of tx_signal by power factor."""
        if self.tx_power != 0:
            power = self.waveform_generator.power

            self.power_factor = self.tx_power / power
            tx_signal = tx_signal * np.sqrt(self.power_factor)

        return tx_signal

    def receive(self, input_signal: np.ndarray, noise_var: float) -> np.ndarray:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            input_signal (np.ndarray): Received signal.
            noise_var (float): noise variance (for equalization).

        Returns:
            np.array: Detected bits as a list of data blocks for the drop.
        """
        rx_signal = self.rf_chain.receive(input_signal)

        # If no receiving waveform generator is configured, no signal is being received
        if self.waveform_generator is None:
            return np.empty(0, dtype=complex)

        # normalize signal to expected input power
        rx_signal = rx_signal / np.sqrt(1.0)  # TODO: Re-implement pair power factor
        noise_var = noise_var / 1.0           # TODO: Re-implement pair power factor

        received_bits = np.empty(0, dtype=int)
        timestamp_in_samples = 0

        while rx_signal.size:
            initial_size = rx_signal.shape[1]
            frame_bits, rx_signal = self.waveform_generator.receive_frame(
                rx_signal, timestamp_in_samples, noise_var)

            if rx_signal.size:
                timestamp_in_samples += initial_size - rx_signal.shape[1]

            received_bits = np.append(received_bits, frame_bits)

        decoded_bits = self.encoder_manager.decode(received_bits)
        return decoded_bits

    def get_bit_energy(self) -> float:
        """Returns the average bit energy of the modulated signal.
        """
        R = self.encoder_manager.rate
        return self.waveform_generator.bit_energy * self.power_factor / R

    def get_symbol_energy(self) -> float:
        """Returns the average symbol energy of the modulated signal.
        """
        R = self.encoder_manager.rate
        return self.waveform_generator.symbol_energy * self.power_factor / R

    @property
    def position(self) -> np.array:
        """Access the modem's position.

        Returns:
            np.array:
                The modem position in xyz-coordinates.
        """

        return self.__position

    @position.setter
    def position(self, position: np.array) -> None:
        """Update the modem's position.

        Args:
            position (np.array):
                The modem's new position.
        """

        self.__position = position

    @property
    def orientation(self) -> np.array:
        """Access the modem's orientation.

        Returns:
            np.array:
                The modem orientation as a normalized quaternion.
        """

        return self.__orientation

    @orientation.setter
    def orientation(self, orientation: np.array) -> None:
        """Update the modem's orientation.

        Args:
            orientation(np.array):
                The new modem orientation.
        """

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

        if len(topology.shape) > 2:
            raise ValueError("The topology array must be of dimension 2")

        if topology.shape[0] < 1:
            raise ValueError("The topology must contain at least one sensor")

        if len(topology.shape) > 1:

            if topology.shape[1] > 3:
                raise ValueError("The second topology dimension must contain 3 fields (xyz)")

            self.__topology = np.zeros((topology.shape[0], 3), dtype=float)
            self.__topology[:, :topology.shape[1]] = topology

        else:

            self.__topology = np.zeros((topology.shape[0], 3), dtype=float)
            self.__topology[:, 0] = topology

        # Automatically detect linearity in default configurations, where all sensor elements
        # are oriented along the local x-axis.
        axis_sums = np.sum(self.__topology, axis=0)

        if (axis_sums[1] + axis_sums[2]) < 1e-10:
            self.__is_linear = True

    @property
    def carrier_frequency(self) -> float:
        """Access the configured carrier frequency of the RF signal.

        Returns:
            float:
                Carrier frequency in Hz.
        """

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, carrier_frequency: float) -> None:
        """Modify the configured center frequency of the steered RF-signal.

        Args:
            carrier_frequency (float):
                Carrier frequency in Hz.

        Raises:
            ValueError:
                If center frequency is less than zero.
        """

        if carrier_frequency < 0.0:
            raise ValueError("Carrier frequency must be greater or equal to zero")

        self.__carrier_frequency = carrier_frequency

    @property
    def sampling_rate(self) -> float:
        """Access the rate at which the analog signals are sampled.

        Returns:
            float:
                Signal sampling rate in Hz.
        """

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        """Modify the rate at which the analog signals are sampled.

        Args:
            value (float):
                Signal sampling rate in Hz.

        Raises:
            ValueError:
                If the sampling rate is less or equal to zero.
        """

        if value <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

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

#    @property
#    def num_streams(self) -> int:
#        """The number of data streams generated by the modem.
#
#        The number of data streams is always less or equal to the number of available antennas `num_antennas`.
#
#        Returns:
#            int:
#                The number of data streams generated by the modem.
#        """
#
#        # For now, only beamforming influences the number of data streams.
#        # This might change in future!
#        return self.__beamformer.num_streams

    @property
    def encoder_manager(self) -> EncoderManager:
        """Access the modem's encoder management.

        Returns:
            EncoderManager:
                Handle to the modem's encoder instance.
        """

        return self.__encoder_manager

    @property
    def bits_source(self) -> BitsSource:
        """Access the modem's configured bits source.

        Returns:
            BitsSource:
                Handle to the modem's bit source instance.
        """

        return self.__bits_source

    @bits_source.setter
    def bits_source(self, bits_source: BitsSource) -> None:
        """Configure the modem's bits source.

        Args:
            bits_source (BitsSource):
                The new bits source.
        """

        self.__bits_source = bits_source

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
    def tx_power(self) -> float:
        """Power of the transmitted signal.

        Returns:
            float: Transmit power.
        """

        return self.__tx_power

    @tx_power.setter
    def tx_power(self, power: float) -> None:
        """Modify the power of the transmitted signal.

        Args:
            power (float): The new signal transmit power in Watts?.

        Raises:
            ValueError: If transmit power is negative.
        """

        if power < 0.0:
            raise ValueError("Transmit power must be greater or equal to zero")

        self.__tx_power = power

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
    def precoding(self) -> Precoding:
        """Access this modem's precoding configuration.

        Returns:
            Precoding: Handle to the configuration.
        """

        return self.__precoding

    @property
    def num_data_bits_per_frame(self) -> int:
        """Compute the number of required data bits to generate a single frame.

        Returns:
            int: The number of data bits.
        """

        num_code_bits = self.waveform_generator.bits_per_frame
        return self.encoder_manager.required_num_data_bits(num_code_bits)
