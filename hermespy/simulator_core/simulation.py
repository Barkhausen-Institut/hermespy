# -*- coding: utf-8 -*-
"""HermesPy simulation configuration."""

from __future__ import annotations
from typing import List, Type, Optional, Union, Tuple
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from .executable import Executable, Verbosity
from .drop import Drop
from .statistics import SNRType, Statistics, ConfidenceMetric
from hermespy.scenario import Scenario
from hermespy.modem import Receiver
from hermespy.channel import QuadrigaInterface, Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulationDrop(Drop):
    """Data generated within a single simulation drop."""

    def __init__(self, *args) -> None:
        """Simulation drop object initialization.
        """

        Drop.__init__(self, *args)



class Simulation(Executable):
    """HermesPy simulation configuration.

    Attributes:

        plot_bit_error (bool):
            Plot resulting bit error rate after simulation.

        plot_block_error (bool):
            Plot resulting block error rate after simulation.

        __min_num_drops (int):
            Minimum number of drops before confidence check may prematurely abort execution.

        __confidence_level (float):
            Confidence at which execution should be terminated.

        __confidence_margin (float):
            Margin for the confidence check
    """

    yaml_tag = u'Simulation'
    snr_type: SNRType
    noise_loop: List[float]
    confidence_metric: ConfidenceMetric
    plot_bit_error: bool
    plot_block_error: bool
    __min_num_drops: int
    __confidence_level: float
    __confidence_margin: float
    plot_drop_transmitted_bits: bool
    plot_drop_transmitted_signals: bool
    plot_drop_received_signals: bool
    plot_drop_received_bits: bool
    plot_drop_bit_errors: bool
    plot_drop_block_errors: bool
    plot_drop_transmit_stft: bool
    plot_drop_receive_stft: bool
    plot_drop_transmit_spectrum: bool
    plot_drop_receive_spectrum: bool

    def __init__(self,
                 plot_drop: bool = True,
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
                 confidence_metric: Union[ConfidenceMetric, str] = ConfidenceMetric.DISABLED,
                 min_num_drops: int = 0,
                 max_num_drops: int = 1,
                 confidence_level: float = 1.0,
                 confidence_margin: float = 0.0,
                 results_dir: Optional[str] = None,
                 verbosity: Union[str, Verbosity] = Verbosity.INFO) -> None:
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

            min_num_drops (int, optional):
                Minimum number of drops before confidence check may prematurely terminate execution.

            max_num_drops (int, optional):
                Maximum number of drops before confidence check may prematurely terminate execution.

            confidence_level (float, optional):
                Confidence at which execution should be terminated.

            confidence_margin (float, optional):
                Margin for the confidence check

            results_dir (str, optional):
                Directory in which all simulation artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.
        """

        Executable.__init__(self, plot_drop, calc_transmit_spectrum, calc_receive_spectrum,
                            calc_transmit_stft, calc_receive_stft, spectrum_fft_size, num_drops,
                            results_dir, verbosity)

        self.plot_drop_transmitted_bits = False
        self.plot_drop_transmitted_signals = False
        self.plot_drop_received_signals = False
        self.plot_drop_received_bits = False
        self.plot_drop_bit_errors = False
        self.plot_drop_block_errors = False
        self.plot_drop_transmit_stft = False
        self.plot_drop_receive_stft = False
        self.plot_drop_transmit_spectrum = False
        self.plot_drop_receive_spectrum = False
        self.snr_type = snr_type
        self.plot_bit_error = plot_bit_error
        self.plot_block_error = plot_block_error
        self.min_num_drops = min_num_drops
        self.max_num_drops = max_num_drops
        self.confidence_level = confidence_level
        self.confidence_margin = confidence_margin

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

        if self.max_num_drops < self.min_num_drops:
            raise ValueError("Minimum number of drops must be smaller than maximum number of drops.")

    def run(self) -> None:
        """Run the full simulation configuration."""

        # Iterate over scenarios
        for s, scenario in enumerate(self.scenarios):

            # Plot scenario information +
            if self.verbosity.value <= Verbosity.INFO.value:

                print(f"\nScenario Simulation #{s}, sampled at {scenario.sampling_rate:.2E}Hz")

                # Warn if the sampling rate is too low
                if self.verbosity.value <= Verbosity.WARNING.value and scenario.sampling_rate < \
                        scenario.min_sampling_rate:
                    print('\033[93m' + "Warning: The chosen sampling rate might be too low" + '\033[0m')

                print(f"{'SNR':<15}{'Drop':<15}{'Link':<15}{'BER':<15}{'FER':<15}")
                print("="*75)

            # Initialize plot statistics with current scenario state
            statistics = Statistics(scenario=scenario,
                                    snr_loop=self.noise_loop,
                                    calc_transmit_spectrum=self.calc_transmit_spectrum,
                                    calc_receive_spectrum=self.calc_receive_spectrum,
                                    calc_transmit_stft=self.calc_transmit_stft,
                                    calc_receive_stft=self.calc_receive_stft,
                                    spectrum_fft_size=self.spectrum_fft_size,
                                    confidence_margin=self.confidence_margin,
                                    confidence_level=self.confidence_level,
                                    confidence_metric=self.confidence_metric,
                                    min_num_drops=self.min_num_drops,
                                    max_num_drops=self.max_num_drops)

            # Save most recent drop
            drop: Optional[SimulationDrop] = None

            for d in range(self.max_num_drops):
                run_flags = statistics.run_flag_matrix

                for noise_index, snr in enumerate(self.noise_loop):
                    drop_run_flag = run_flags[:, :, noise_index]
                    # Generate data bits to be transmitted
                    data_bits = scenario.generate_data_bits()

                    # Generate radio-frequency band signal emitted from each transmitter
                    transmitted_signals = Simulation.transmit(scenario=scenario,
                                                              drop_run_flag=drop_run_flag,
                                                              data_bits=data_bits)

                    # Simulate propagation over channel model
                    propagation_matrix = Simulation.propagate(scenario,
                                                              transmitted_signals,
                                                              drop_run_flag)

                    # Simulate signal reception and mixing at the receiver-side
                    received_signals = Simulation.receive(scenario,
                                                          propagation_matrix,
                                                          drop_run_flag,
                                                          self.snr_type,
                                                          snr)

                    # Receive and demodulate signal
                    detected_bits = Simulation.detect(scenario,
                                                      received_signals,
                                                      drop_run_flag)

                    # Collect block sizes
                    transmit_block_sizes = scenario.transmit_block_sizes
                    receive_block_sizes = scenario.receive_block_sizes

                    # ToDo: Maybe change the structure here
                    received_samples = [received_signal[0] for received_signal in received_signals]

                    # Save generated signals
                    drop = SimulationDrop(data_bits, transmitted_signals, transmit_block_sizes, received_samples,
                                          detected_bits, receive_block_sizes, True, self.spectrum_fft_size,
                                          scenario.sampling_rate)

                    # Print drop statistics if verbosity flag is set
                    if self.verbosity.value <= Verbosity.INFO.value:

                        bers = drop.bit_error_rates
                        blers = drop.block_error_rates

                        for (tx_id, tx_bers), tx_blers in zip(enumerate(bers), blers):
                            for (rx_id, ber), bler in zip(enumerate(tx_bers), tx_blers):

                                link_str = f"{tx_id}x{rx_id}"
                                ber_str = "-" if ber is None else f"{bler:.4f}"
                                bler_str = "-" if bler is None else f"{bler:.4f}"

                                if tx_id == 0 and rx_id == 0:

                                    snr_str = f"{10 * np.log10(snr):.1f}"
                                    print(f"{snr_str:<15}{d:<15}{link_str:<15}{ber_str:<15}{bler_str:<15}")

                                else:
                                    print(" " * 30 + f"{link_str:<15}{ber_str:<15}{bler_str:<15}")

                    # Visualize plot if requested
                    if self.plot_drop:

                        if self.plot_drop_transmitted_bits:
                            drop.plot_transmitted_bits()

                        if self.plot_drop_transmitted_signals:
                            drop.plot_transmitted_signals()

                        if self.plot_drop_received_signals:
                            drop.plot_received_signals()

                        if self.plot_drop_received_bits:
                            drop.plot_received_bits()

                        if self.plot_drop_bit_errors:
                            drop.plot_bit_errors()

                        if self.plot_drop_block_errors:
                            drop.plot_block_errors()

                        if self.plot_drop_transmit_stft:
                            drop.plot_transmit_stft()

                        if self.plot_drop_receive_stft:
                            drop.plot_receive_stft()

                        if self.plot_drop_transmit_spectrum:
                            drop.plot_transmit_spectrum()

                        if self.plot_drop_receive_spectrum:
                            drop.plot_receive_spectrum()

                        plt.show()

                    # Add drop to the statistics
                    statistics.add_drop(drop, noise_index)

                    # Check confidence if the routine is enabled and
                    # the minimum number of configured drops has been reached
                    # if self.confidence_metric is not ConfidenceMetric.DISABLED and d >= self.min_num_drops:

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

    @property
    def max_num_drops(self) -> int:
        """Maximum number of drops before confidence check may terminate execution.

        Returns:
            int: Maximum number of drops.
        """
        return self.__max_num_drops

    @max_num_drops.setter
    def max_num_drops(self, val: int) -> None:
        """Modify maximum number of drops before confidence check may terminate execution.

        Args:
            num_drops (int): maximum number of drops.

        Raises:
            ValueError: If `num_drops` is smaller than zero.
        """

        if val < 0:
            raise ValueError("Maximum number of drops must be greater or equal to zero.")

        self.__max_num_drops = val

    @property
    def min_num_drops(self) -> int:
        """Minimum number of drops before confidence check may terminate execution.

        Returns:
            int: Minimum number of drops.
        """

        return self.__min_num_drops

    @min_num_drops.setter
    def min_num_drops(self, val: int) -> None:
        """Modify minimum number of drops before confidence check may terminate execution.

        Args:
            num_drops (int): Minim number of drops.

        Raises:
            ValueError: If `num_drops` is smaller than zero.
        """

        if val < 0:
            raise ValueError("Minimum number of drops must be greater or equal to zero.")

        self.__min_num_drops = val

    @property
    def confidence_level(self) -> float:
        """Access confidence level at which execution may be prematurely terminated.

        Return:
            float: Confidence level between 0.0 and 1.0.
        """

        return self.__confidence_level

    @confidence_level.setter
    def confidence_level(self, level: float) -> None:
        """Modify confidence level at which execution may be prematurely terminated.

        Args:
            level (float): Confidence level between 0.0 and 1.0.

        Raises:
            ValueError: If `level` is not between 0.0 and 1.0.
        """

        if not 0.0 <= level <= 1.0:
            raise ValueError("Confidence level must be between zero and one")

        self.__confidence_level = level
        
    @property
    def confidence_margin(self) -> float:
        """Access margin for confidence level at which execution may be prematurely terminated.

        Return:
            float: Absolute margin confidence margin.
        """

        return self.__confidence_margin

    @confidence_margin.setter
    def confidence_margin(self, margin: float) -> None:
        """Modify margin for confidence level at which execution may be prematurely terminated.

        Args:
            margin (float): Absolute margin.

        Raises:
            ValueError: If `margin` is smaller than zero.
        """

        if margin < 0.0:
            raise ValueError("Margin must be greater or equal to zero")

        self.__confidence_margin = margin

    @staticmethod
    def transmit(scenario: Scenario,
                 drop_run_flag: np.ndarray,
                 drop_duration: Optional[float] = None,
                 data_bits: Optional[np.array] = None) -> List[np.ndarray]:
        """Simulate signals emitted by all transmitters registered with a scenario.

        Args:
            scenario (Scenario): The scenario for which to simulate signals.
            drop_run_flag (np.ndarray): Mask that says if signals are to be created for specific snr.
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

        sending_tx_idx = np.flatnonzero(np.all(drop_run_flag, axis=1))

        transmitted_signals = []
        if data_bits is None:

            for transmitter_idx, transmitter in enumerate(scenario.transmitters):
                if transmitter_idx in sending_tx_idx:
                    transmitted_signals.append(transmitter.send(drop_duration))
                else:
                    transmitted_signals.append(None)

        else:

            if len(data_bits) != len(scenario.transmitters):
                raise ValueError("Data bits to be transmitted contain insufficient streams")
            
            for transmitter_idx, (transmitter, data) in enumerate(
                                                            zip(scenario.transmitters, data_bits)):
                if transmitter_idx in sending_tx_idx:
                    transmitted_signals.append(transmitter.send(drop_duration, data))
                else:
                    transmitted_signals.append(None)

        return transmitted_signals

    @staticmethod
    def propagate(scenario: Scenario,
                  transmitted_signals: List[np.ndarray],
                  drop_run_flag: np.ndarray) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension

        Args:
            scenario (Scenario): The scenario for which to simulate the channel propagation.
            transmitted_signals (List[np.ndarray]):
                List of signal streams emerging from each registered transmit modem.
            drop_run_flag (np.ndarray): Mask that says if signals are to be created for specific snr.

        Returns:
            List[List[Tuple[np.ndarray, np.ndarray]]]:
                MxN Matrix of pairs of received signals and impulse responses.
                The entry in the M-th row and N-th column contains the propagation data between
                the N-th transmitter and M-th receiver.

        Raises:
            ValueError: If the number of `transmitted_signals` does not equal the number of registered transmit modems.
        """

        if len(transmitted_signals) != len(scenario.transmitters):
            raise ValueError("Number of transmit signals {} does not match the number of registered transmit "
                             "modems {}".format(len(transmitted_signals), len(scenario.transmitters)))

        # Access the channel models
        channels = scenario.channels

        # Initialize the propagated signals
        propagation_matrix: List[List[Tuple[np.ndarray, np.ndarray]]] = []

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        for receiver_id, receiver in enumerate(scenario.receivers):

            propagation_row: List[Tuple[np.ndarray, np.ndarray]] = []
            for transmitter_id, (transmitter, transmitted_signal) in enumerate(zip(scenario.transmitters,
                                                                                   transmitted_signals)):
                propagation_tuple = tuple((None, None))
                # Select responsible channel between respective transmitter and receiver
                channel: Channel = channels[transmitter_id, receiver_id]
                propagate_signal = drop_run_flag[transmitter_id, receiver_id]
                # Propagate the signal over the channel
                # The propagation tuple contains the propagated signal as the first element,
                # the impulse response as the second element
                if propagate_signal:
                    propagation_tuple = channel.propagate(transmitted_signal)
                propagation_row.append(propagation_tuple)

            propagation_matrix.append(propagation_row)

        return propagation_matrix

    @staticmethod
    def receive(scenario: Scenario,
                propagation_matrix: List[List[Tuple[np.ndarray, np.ndarray]]],
                drop_run_flag: np.ndarray,
                snr_type: SNRType = SNRType.EBN0,
                snr: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate signals received by all receivers registered with this scenario.

        This routine superimposes propagated signals at the receiver-inputs,
        introduces additive thermal noise and mixes according to carrier frequency configurations.

        Args:
            scenario (Scenario):
                The scenario for which to simulate the received signals.

            propagation_matrix (List[List[Tuple[np.ndarray, np.ndarray]]]):
                MxN Matrix of pairs of received signals and impulse responses.
                The entry in the M-th row and N-th column contains the propagation data between
                the N-th transmitter and M-th receiver.

            drop_run_flag (np.ndarray): Mask that says if signals are to be created for specific snr.

            snr_type (SNRType, optional):
                Type of noise.

            snr (float, optional):
                Signal to noise ratio.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]:
                A list of M tuples containing the received noisy base-band signals, the channel impulse response
                as well as the noise variance for each receiving modem, respectively.

        Raises:
            ValueError:
                If the number of `arriving_signals` does not equal the number of registered receive modems.
        """

        if len(propagation_matrix) != len(scenario.receivers):
            raise ValueError("Number of arriving signals {} does not match the number of registered receive "
                             "modems {}".format(len(propagation_matrix), len(scenario.receivers)))

        # Prepare the receive signals
        received_signals: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for receiver_index, (receiver, transmitted_signals) in enumerate(zip(scenario.receivers,
                                                                             propagation_matrix)):
            impinging_signals = [(tx_tuple[0], transmitter.carrier_frequency)
                                 for tx_tuple, transmitter in zip(transmitted_signals, scenario.transmitters)]
            # delete masked signals
            receiving_tx = np.flatnonzero(drop_run_flag[:, receiver_index])
            if len(receiving_tx) > 0:
                impinging_signals = [impinging_signals[tx_idx] for tx_idx in receiving_tx]

                # Compute noise variance at the receiver
                noise_variance = Simulation.calculate_noise_variance(
                    receiver, snr, snr_type)

                # Model rf-signal reception at the respective receiver
                received_signal = receiver.receive(impinging_signals, noise_variance)

                # Select the proper channel impulse response
                channel = transmitted_signals[receiver.reference_transmitter.index][1]

                # Save result, what else?
                received_signals.append((received_signal, channel, noise_variance))
            else:
                received_signals.append((None, None, None))

        return received_signals

    @staticmethod
    def calculate_noise_variance(receiver: Receiver, snr: float, snr_type: SNRType) -> float:
        if snr_type == SNRType.EBN0:
            noise_variance = receiver.waveform_generator.bit_energy / snr

        elif snr_type == SNRType.ESN0:
            noise_variance = receiver.waveform_generator.symbol_energy / snr

        elif snr_type == SNRType.CUSTOM:  # TODO: What is custom exactly supposed to do?
            noise_variance = 1 / snr
        return noise_variance


    @staticmethod
    def detect(scenario: Scenario,
               received_signals: List[Tuple[np.ndarray, np.ndarray, float]],
               drop_run_flag: np.ndarray) -> List[np.ndarray]:
        """Detect bits from base-band signals.

        Calls the waveform-generator's receive-chain routines of each respective receiver.

        Args:
            scenario (Scenario):
                The scenario for which to simulate bit detection.

            received_signals (List[Tuple[np.ndarray, np.ndarray, float]]):
                A list of M tuples containing the received noisy base-band signals, the channel impulse response
                as well as the noise variance for each receiving modem, respectively.

            drop_run_flag (np.ndarray): Mask that says if signals are to be created for specific snr.

        Returns:
            List[np.ndarray]:
                A list of M bit streams, where the m-th entry contains the bits detected by the m-th
                receiving modem within the m-th baseband signal.

        Raises:
            ValueError:
                If `received_signals` contains less entries than receivers registered in `scenario`.
        """

        if scenario.num_receivers != len(received_signals):
            raise ValueError("Less received signals than scenario receivers provided")

        receiver_bits: List[np.ndarray] = []

        for receiver, (signal, channel, noise) in zip(scenario.receivers, received_signals):
            bits = None
            active_rx_idx = np.flatnonzero(np.all(drop_run_flag, axis=0))

            if receiver.index in active_rx_idx:
                bits = receiver.demodulate(signal, channel, noise)
            receiver_bits.append(bits)

        return receiver_bits

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
            "noise_loop": node.noise_loop,
            "verbosity": node.verbosity.name
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
            Simulation:
                Newly created `Simulation` instance.
        """

        state = constructor.construct_mapping(node)

        # Launch a global quadriga instance
        quadriga_interface: Optional[QuadrigaInterface] = state.pop(QuadrigaInterface.yaml_tag, None)
        if quadriga_interface is not None:
            QuadrigaInterface.SetGlobalInstance(quadriga_interface)

        plot_drop_transmitted_bits = state.pop('plot_drop_transmitted_bits', False)
        plot_drop_transmitted_signals = state.pop('plot_drop_transmitted_signals', False)
        plot_drop_received_signals = state.pop('plot_drop_received_signals', False)
        plot_drop_received_bits = state.pop('plot_drop_received_bits', False)
        plot_drop_bit_errors = state.pop('plot_drop_bit_errors', False)
        plot_drop_block_errors = state.pop('plot_drop_block_errors', False)
        plot_drop_transmit_stft = state.pop('plot_drop_transmit_stft', False)
        plot_drop_receive_stft = state.pop('plot_drop_receive_stft', False)
        plot_drop_transmit_spectrum = state.pop('plot_drop_transmit_spectrum', False)
        plot_drop_receive_spectrum = state.pop('plot_drop_receive_spectrum', False)

        # Convert noise loop dB to linear
        noise_loop = state.pop('noise_loop', None)
        if noise_loop is not None:
            state['noise_loop'] = 10 ** (np.array(noise_loop) / 10)

        simulation = cls(**state)

        simulation.plot_drop_transmitted_bits = plot_drop_transmitted_bits
        simulation.plot_drop_transmitted_signals = plot_drop_transmitted_signals
        simulation.plot_drop_received_signals = plot_drop_received_signals
        simulation.plot_drop_received_bits = plot_drop_received_bits
        simulation.plot_drop_bit_errors = plot_drop_bit_errors
        simulation.plot_drop_block_errors = plot_drop_block_errors
        simulation.plot_drop_transmit_stft = plot_drop_transmit_stft
        simulation.plot_drop_receive_stft = plot_drop_receive_stft
        simulation.plot_drop_transmit_spectrum = plot_drop_transmit_spectrum
        simulation.plot_drop_receive_spectrum = plot_drop_receive_spectrum

        return simulation
