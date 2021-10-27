# -*- coding: utf-8 -*-
"""HermesPy simulation configuration."""

from __future__ import annotations
from typing import List, Type, Optional, Union
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from .executable import Executable
from .drop import Drop
from .statistics import SNRType, Statistics
from scenario import Scenario
from channel import QuadrigaInterface, Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulationDrop(Drop):
    """Data generated within a single simulation drop."""

    def __init__(self, *args) -> None:
        """Simulation drop object initialization.
        """

        Drop.__init__(self, *args)


class ConfidenceMetric(Enum):
    """Confidence metric for stopping criteria during simulation execution. """

    DISABLED = 0        # No stopping criterion
    BER = 1             # Bit Error Rate
    FER = 2             # ? Error Rate


class Simulation(Executable):
    """HermesPy simulation configuration.

    Attributes:

        plot_bit_error (bool):
            Plot resulting bit error rate after simulation.

        plot_block_error (bool):
            Plot resulting block error rate after simulation.
    """

    yaml_tag = u'Simulation'
    snr_type: SNRType
    noise_loop: List[float]
    confidence_metric: ConfidenceMetric
    plot_bit_error: bool
    plot_block_error: bool

    def __init__(self,
                 plot_drop: bool = False,
                 calc_transmit_spectrum: bool = False,
                 calc_receive_spectrum: bool = False,
                 calc_transmit_stft: bool = False,
                 calc_receive_stft: bool = False,
                 spectrum_fft_size: int = 0,
                 num_drops: int = 1,
                 plot_bit_error: bool = True,
                 plot_block_error: bool = True,
                 snr_type: Union[str, SNRType] = SNRType.EBN0,
                 noise_loop: Union[List[float], np.ndarray] = np.array([0.0]),
                 confidence_metric: Union[ConfidenceMetric, str] = ConfidenceMetric.DISABLED) -> None:
        """Simulation object initialization.

        Args:
            plot_drop (bool, optional):
                Plot each drop during execution of scenarios.

            calc_transmit_spectrum (bool):
                Compute the transmitted signals frequency domain spectra.

            calc_receive_spectrum (bool):
                Compute the received signals frequency domain spectra.

            calc_transmit_stft (bool):
                Compute the short time Fourier transform of transmitted signals.

            calc_receive_stft (bool):
                Compute the short time Fourier transform of received signals.

            spectrum_fft_size (int):
                Number of discrete frequency bins computed within the Fast Fourier Transforms.

            num_drops (int):
                Number of drops per executed scenario.

            plot_bit_error (bool, optional):
                Plot resulting bit error rate after simulation.

            plot_block_error (bool, optional):
                Plot resulting block error rate after simulation.

            snr_type (Union[str, SNRType]):
                The signal to noise ratio metric to be used.

            noise_loop (Union[List[float], np.ndarray]):
                Loop over different noise levels.

            confidence_metric (Union[ConfidenceMetric, str], optional):
                Metric for premature simulation stopping criterion
        """

        Executable.__init__(self, plot_drop, calc_transmit_spectrum, calc_receive_spectrum,
                            calc_transmit_stft, calc_receive_stft, spectrum_fft_size, num_drops)

        self.snr_type = snr_type
        self.plot_bit_error = plot_bit_error
        self.plot_block_error = plot_block_error

        # Convert noise loop from array to list if the provided argument is a numpy array
        if isinstance(noise_loop, np.ndarray):
            self.noise_loop = noise_loop.tolist()

        else:
            self.noise_loop = noise_loop

        # Recover confidence metric enumeration from string value if the provided argument is a string
        if isinstance(confidence_metric, str):
            self.confidence_metric = ConfidenceMetric[confidence_metric]

        else:
            self.confidence_metric = confidence_metric

    def run(self) -> None:
        """Run the full simulation configuration."""

        # Iterate over scenarios
        for scenario in self.scenarios:

            # Initialize plot statistics with current scenario state
            snr = (10 * (np.log10(np.array(self.noise_loop)))).tolist()
            statistics = Statistics(scenario, snr, self.calc_transmit_spectrum, self.calc_receive_spectrum,
                                    self.calc_transmit_stft, self.calc_receive_stft, self.spectrum_fft_size)

            # Save most recent drop
            drop: Optional[SimulationDrop] = None

            # Iterate over required noise taps
            for noise_index, snr in enumerate(self.noise_loop):

                # Iterate over configured number of required simulation drops
                for d in range(self.num_drops):

                    # Generate data bits to be transmitted
                    data_bits = scenario.generate_data_bits()

                    # Generate radio-frequency band signal emitted from each transmitter
                    transmitted_signals = Simulation.transmit(scenario, data_bits=data_bits)

                    # Simulate propagation over channel model
                    propagated_signals = Simulation.propagate(scenario, transmitted_signals)

                    # Receive and demodulate signal
                    received_bits = Simulation.receive(scenario, propagated_signals, self.snr_type, snr)

                    # Collect block sizes
                    transmit_block_sizes = scenario.transmit_block_sizes
                    receive_block_sizes = scenario.receive_block_sizes

                    # Save generated signals
                    drop = SimulationDrop(data_bits, transmitted_signals, transmit_block_sizes, propagated_signals,
                                          received_bits, receive_block_sizes)

                    # Visualize plot if requested
                    if self.plot_drop:

                        drop.plot()
                        plt.show()

                    # Add drop to the statistics
                    statistics.add_drop(drop, noise_index)

            # Dump statistics results
            statistics.save(self.results_dir)

            # Plot statistics results, if flags apply
            if self.calc_transmit_spectrum:
                statistics.plot_transmit_spectrum()

            if self.calc_receive_spectrum:
                statistics.plot_receive_spectrum()

            if self.plot_bit_error:
                statistics.plot_bit_error_rates()

            if self.plot_block_error:
                statistics.plot_block_error_rates()

            # Plot the last drop to visualize stfts, if available
            if drop is not None:

                if self.calc_transmit_stft:
                    drop.plot_transmit_stft()

                if self.calc_receive_stft:
                    drop.plot_receive_stft()

            plt.show()

    @property
    def snr_type(self) -> SNRType:
        """Type of signal-to-noise ratio.

        Returns:
            SNRType: The SNR type.
        """

        return self.__snr_type

    @snr_type.setter
    def snr_type(self, snr_type: Union[str, SNRType]) -> None:
        """Modify the type of signal-to-noise ratio.

        Args:
            snr_type (Union[str, SNRType]):
                The new type of signal to noise ratio, string or enum representation.
        """

        if isinstance(snr_type, str):
            snr_type = SNRType[snr_type]

        self.__snr_type = snr_type

    @property
    def noise_loop(self) -> List[float]:
        """Access the configured signal to noise ratios over which the simulation iterates.

        Returns:
            np.ndarray: The signal to noise ratios.
        """

        return self.__noise_loop

    @noise_loop.setter
    def noise_loop(self, loop: Union[List[float], np.ndarray]) -> None:
        """Modify the configured signal to noise ratios over which the simulation iterates.
        
        Args:
            loop (Union[List[float], np.ndarray]): The new noise loop.

        Raises:
            ValueError: If `loop` does not represent a vector with at least one entry.
        """

        # Convert arrays to list
        if isinstance(loop, np.ndarray):

            if loop.ndim != 1:
                raise ValueError("The noise loop must be a vector")

            loop = loop.tolist()

        if len(loop) < 1:
            raise ValueError("The noise loop must contain at least one SNR entry")

        self.__noise_loop = loop

    @staticmethod
    def transmit(scenario: Scenario,
                 drop_duration: Optional[float] = None,
                 data_bits: Optional[np.array] = None) -> List[np.ndarray]:
        """Simulate signals emitted by all transmitters registered with a scenario.

        Args:
            scenario (Scenario): The scenario for which to simulate signals.
            drop_duration (float, optional): Length of simulated transmission in seconds.
            data_bits (List[np.array], optional): The data bits to be sent by each transmitting modem.

        Returns:
            List[np.ndarray]: A list containing the the signals emitted by each transmitting modem.

        Raises:
            ValueError: On invalid `drop_duration`s.
            ValueError: If `data_bits` does not contain data for each transmitting modem.
        """

        if drop_duration is None:
            drop_duration = scenario.drop_duration

        if drop_duration <= 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        transmitted_signals = []

        if data_bits is None:

            for transmitter in scenario.transmitters:
                transmitted_signals.append(transmitter.send(drop_duration))

        else:

            if len(data_bits) != len(scenario.transmitters):
                raise ValueError("Data bits to be transmitted contain insufficient streams")

            for transmitter, data in zip(scenario.transmitters, data_bits):
                transmitted_signals.append(transmitter.send(drop_duration, data))

        return transmitted_signals

    @staticmethod
    def propagate(scenario: Scenario,
                  transmitted_signals: List[np.ndarray]) -> List[np.ndarray]:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension

        Args:
            scenario (Scenario): The scenario for which to simulate the channel propagation.
            transmitted_signals (List[np.ndarray]):
                List of signal streams emerging from each registered transmit modem.

        Returns:
            List[np.ndarray]:
                List of propagated signal streams impinging onto the registered receive modems.

        Raises:
            ValueError: If the number of `transmitted_signals` does not equal the number of registered transmit modems.
        """

        if len(transmitted_signals) != len(scenario.transmitters):
            raise ValueError("Number of transmit signals {} does not match the number of registered transmit "
                             "modems {}".format(len(transmitted_signals), len(scenario.transmitters)))

        # Access the channel models
        channels = scenario.channels

        # Initialize the propagated signals
        # noinspection PyTypeChecker
        arriving_signals = [np.empty((receiver.num_antennas, 0), dtype=complex) for receiver in scenario.receivers]

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        for transmitter_id, transmitted_signal in enumerate(transmitted_signals):
            for receiver_id, receiver in enumerate(scenario.receivers):

                # Select responsible channel between respective transmitter and receiver
                channel: Channel = channels[transmitter_id, receiver_id]

                # Skip propagation over channels flagged as inactive
                if not channel.active:
                    continue

                # Propagate signal emerging from transmitter over the channel
                propagated_signal = channel.propagate(transmitted_signal)

                # Extend the propagated signals matrix to hold more samples (if required
                sample_difference = propagated_signal.shape[1] - arriving_signals[receiver_id].shape[1]
                if sample_difference > 0:

                    # noinspection PyTypeChecker
                    arriving_signals[receiver_id] = np.append(arriving_signals[receiver_id],
                                                              np.zeros((receiver.num_antennas, sample_difference),
                                                                       dtype=complex), axis=1)
                    arriving_signals[receiver_id] += propagated_signal

                elif sample_difference < 0:
                    arriving_signals[receiver_id][:, :sample_difference] += propagated_signal

                else:
                    arriving_signals[receiver_id] += propagated_signal

        return arriving_signals

    @staticmethod
    def receive(scenario: Scenario,
                arriving_signals: List[np.ndarray],
                snr_type: SNRType = SNRType.EBN0,
                snr: float = 0.0) -> List[np.ndarray]:
        """Generate signals received by all receivers registered with this scenario.

        Args:
            scenario (Scenario): The scenario for which to simulate the received signals.
            arriving_signals (List[np.ndarray]): List of signal streams arriving at each receiving modem.
            snr_type (SNRType, optional): Type of noise.
            snr (float, optional): Noise.

        Returns:
            List[np.ndarray]: A list containing the the signals emitted by each transmitting modem.

        Raises:
            ValueError:
                If the number of `arriving_signals` does not equal the number of registered receive modems.
        """

        if len(arriving_signals) != len(scenario.receivers):
            raise ValueError("Number of arriving signals {} does not match the number of registered receive "
                             "modems {}".format(len(arriving_signals), len(scenario.receivers)))

        data_bits: List[np.ndarray] = []

        for receiver_index, receiver in enumerate(scenario.receivers):

            # Compute noise variance at the receiver
            noise_variance = 0.0

            if snr_type == SNRType.EBN0:
                noise_variance = receiver.waveform_generator.bit_energy / snr

            elif snr_type == SNRType.ESN0:
                noise_variance = receiver.waveform_generator.symbol_energy / snr

            elif snr_type == SNRType.CUSTOM:  # TODO: What is custom exactly supposed to do?
                noise_variance = snr

            # Receive data bits
            data = receiver.receive(arriving_signals[receiver_index], noise_variance)
            data_bits.append(data)

        return data_bits

    @classmethod
    def to_yaml(cls: Type[Simulation],
                representer: SafeRepresenter,
                node: Simulation) -> MappingNode:
        """Serialize an `Simulation` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Simulation):
                The `Simulation` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "plot_drop": node.plot_drop,
            "snr_type": node.snr_type.value,
            "noise_loop": node.noise_loop
        }

        # If a global Quadriga interface exists,
        # add its configuration to the simulation section
        if QuadrigaInterface.GlobalInstanceExists():
            state[QuadrigaInterface.yaml_tag] = QuadrigaInterface.GlobalInstance()

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[Simulation],
                  constructor: SafeConstructor,
                  node: MappingNode) -> Simulation:
        """Recall a new `Simulation` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Simulation` serialization.

        Returns:
            WaveformGenerator:
                Newly created `Simulation` instance.
        """

        state = constructor.construct_mapping(node)

        # Launch a global quadriga instance
        quadriga_interface: Optional[QuadrigaInterface] = state.pop(QuadrigaInterface.yaml_tag, None)
        if quadriga_interface is not None:
            QuadrigaInterface.SetGlobalInstance(quadriga_interface)

        # Convert noise loop dB to linear
        noise_loop = state.pop('noise_loop', None)
        if noise_loop is not None:
            state['noise_loop'] = 10 ** (np.array(noise_loop) / 10)

        return cls(**state)
