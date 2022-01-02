# -*- coding: utf-8 -*-
"""HermesPy simulation configuration."""

from __future__ import annotations
from math import ceil, floor
from typing import Any, List, Type, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from ray import remote
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from hermespy.core.executable import Executable, Verbosity
from hermespy.core.drop import Drop
from hermespy.channel import QuadrigaInterface, Channel, ChannelStateInformation

from hermespy.core.factory import Serializable
from hermespy.core.monte_carlo import MonteCarlo, MonteCarloActor, MO
from hermespy.core.scenario import Scenario
from hermespy.core.signal_model import Signal
from hermespy.core.statistics import SNRType, Statistics, ConfidenceMetric
from .simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulationDrop(Drop):
    """Data generated within a single simulation drop."""

    def __init__(self, *args) -> None:
        """Simulation drop object initialization.
        """

        Drop.__init__(self, *args)


class Simulation(Executable, Scenario[SimulatedDevice], Serializable, MonteCarlo[Scenario[SimulatedDevice]]):
    """HermesPy simulation configuration.

    """

    yaml_tag = u'Simulation'

    __channels: np.ndarray
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
    plot_drop_transmitted_symbols: bool
    plot_drop_received_symbols: bool
    plot_drop_transmit_stft: bool
    plot_drop_receive_stft: bool
    plot_drop_transmit_spectrum: bool
    plot_drop_receive_spectrum: bool
    __snr: Optional[float]

    def __init__(self,
                 drop_duration: float = 0.,
                 plot_drop: bool = True,
                 calc_transmit_spectrum: bool = False,
                 calc_receive_spectrum: bool = False,
                 calc_transmit_stft: bool = False,
                 calc_receive_stft: bool = False,
                 spectrum_fft_size: int = 0,
                 plot_bit_error: bool = False,
                 plot_block_error: bool = False,
                 plot_drop_transmitted_symbols: bool = False,
                 plot_drop_received_symbols: bool = False,
                 snr_type: Union[str, SNRType] = SNRType.EBN0,
                 confidence_metric: Union[ConfidenceMetric, str] = ConfidenceMetric.DISABLED,
                 min_num_drops: int = 0,
                 max_num_drops: int = 1,
                 confidence_level: float = 1.0,
                 confidence_margin: float = 0.0,
                 results_dir: Optional[str] = None,
                 verbosity: Union[str, Verbosity] = Verbosity.INFO,
                 seed: Optional[int] = None) -> None:
        """Simulation object initialization.

        Args:

            drop_duration(float, optional):
                Duration of simulation drops in seconds.

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

            plot_bit_error (bool, optional):
                Plot resulting bit error rate after simulation.

            plot_block_error (bool, optional):
                Plot resulting block error rate after simulation.

            plot_drop_transmitted_symbols (bool, optional):
                Plot the constellation of transmitted symbols

            plot_drop_received_symbols (bool, optional);
                Plot the constellation of received symbols.

            snr_type (Union[str, SNRType]):
                The signal to noise ratio metric to be used.

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

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        Executable.__init__(self, plot_drop, calc_transmit_spectrum, calc_receive_spectrum,
                            calc_transmit_stft, calc_receive_stft, spectrum_fft_size, max_num_drops,
                            results_dir, verbosity)

        Scenario.__init__(self, seed=seed)
        MonteCarlo.__init__(self, investigated_object=self, num_samples=max_num_drops)

        self.__channels = np.ndarray((0, 0), dtype=object)
        self.drop_duration = drop_duration
        self.plot_drop_transmitted_bits = False
        self.plot_drop_transmitted_signals = False
        self.plot_drop_received_signals = False
        self.plot_drop_received_bits = False
        self.plot_drop_bit_errors = False
        self.plot_drop_block_errors = False
        self.plot_drop_transmitted_symbols = plot_drop_transmitted_symbols
        self.plot_drop_received_symbols = plot_drop_received_symbols
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
        self.snr = None

        # Recover confidence metric enumeration from string value if the provided argument is a string
        if isinstance(confidence_metric, str):
            self.confidence_metric = ConfidenceMetric[confidence_metric]

        else:
            self.confidence_metric = confidence_metric

        if self.max_num_drops < self.min_num_drops:
            raise ValueError("Minimum number of drops must be smaller than maximum number of drops.")

    def new_device(self) -> SimulatedDevice:
        """Add a new device to the simulation scenario.

        Returns:
            SimulatedDevice: Newly added simulated device.
        """

        device = SimulatedDevice()
        self.add_device(device)

        return device

    def add_device(self, device: SimulatedDevice) -> None:

        # Add the device to the scenario
        Scenario.add_device(self, device)
        device.scenario = self

        if self.num_devices == 1:

            self.__channels = np.array([[Channel(device, device)]], dtype=object)

        else:

            self.__channels = np.append(self.__channels,
                                        np.array([[None for _ in self.devices[:-1]]]), axis=1)
            self.__channels = np.append(self.__channels,
                                        np.array([[Channel(device, rx) for rx in self.devices]]), axis=0)

    @property
    def channels(self) -> np.ndarray:
        """Channel matrix between devices.

        Returns:
            np.ndarray:
                An `MxM` matrix of channels between devices.
        """

        return self.__channels

    def channel(self,
                transmitter: SimulatedDevice,
                receiver: SimulatedDevice) -> Channel:
        """Access a specific channel between two devices.

        Args:

            transmitter (SimulatedDevice):
                The device transmitting into the channel.

            receiver (SimulatedDevice):
                the device receiving from the channel

        Returns:
            Channel:
                Channel between `transmitter` and `receiver`.

        Raises:
            ValueError:
                Should `transmitter` or `receiver` not be registered with this scenario.
        """

        devices = self.devices

        if transmitter not in devices:
            raise ValueError("Provided transmitter is not registered with this scenario")

        if receiver not in devices:
            raise ValueError("Provided receiver is not registered with this scenario")

        index_transmitter = devices.index(transmitter)
        index_receiver = devices.index(receiver)

        return self.__channels[index_transmitter, index_receiver]

    def departing_channels(self, transmitter: SimulatedDevice, active_only: bool = False) -> List[Channel]:
        """Collect all channels departing from a transmitting device.

        Args:

            transmitter (SimulatedDevice):
                The transmitting device.

            active_only (bool, optional):
                Consider only active channels.

        Returns:
            List[Channel]:
                A list of departing channels.

        Raises:
            ValueError:
                Should `transmitter` not be registered with this scenario.
        """

        devices = self.devices

        if transmitter not in devices:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        transmitter_index = devices.index(transmitter)
        channels: List[Channel] = self.__channels[transmitter_index, :].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def arriving_channels(self, receiver: SimulatedDevice, active_only: bool = False) -> List[Channel]:
        """Collect all channels arriving at a device.

        Args:
            receiver (Receiver):
                The receiving modem.

            active_only (bool, optional):
                Consider only active channels.

        Returns:
            List[Channel]:
                A list of arriving channels.

        Raises:
            ValueError:
                Should `receiver` not be registered with this scenario.
        """

        devices = self.devices

        if receiver not in devices:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        receiver_index = devices.index(receiver)
        channels: List[Channel] = self.__channels[:, receiver_index].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def set_channel(self, receiver_index: int, transmitter_index: int, channel: Channel) -> None:
        """Specify a channel within the channel matrix.

        Args:

            receiver_index (int):
                Index of the receiver within the channel matrix.

            transmitter_index (int):
                Index of the transmitter within the channel matrix.

            channel (Channel):
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).

        Raises:
            ValueError:
                If `transmitter_index` or `receiver_index` are greater than the channel matrix dimensions.
        """

        if self.__channels.shape[0] <= transmitter_index or 0 > transmitter_index:
            raise ValueError("Transmitter index greater than channel matrix dimension")

        if self.__channels.shape[1] <= receiver_index or 0 > receiver_index:
            raise ValueError("Receiver index greater than channel matrix dimension")

        # Update channel field within the matrix
        self.__channels[transmitter_index, receiver_index] = channel

        # Set proper receiver and transmitter fields
        channel.transmitter = self.devices[transmitter_index]
        channel.receiver = self.devices[receiver_index]
        channel.random_mother = self
        channel.scenario = self

    def run(self) -> None:

        self.simulate(SimulationActor)

#    def run(self) -> Statistics:
#        """Run the full simulation configuration.
#
#        Returns:
#            Statistics: Statistics of the simulation.
#        """
#
#        # Plot scenario information
#        if self.verbosity.value <= Verbosity.INFO.value:
#
#            print(f"\nExecuting scenario simulation")
#            print(f"{'SNR':<10}{'Drop':<10}{'Link':<15}{'BER':<15}{'BLER':<15}")
#            print("="*65)
#
#        # Initialize plot statistics with current scenario state
#        statistics = Statistics(scenario=self,
#                                snr_loop=self.noise_loop,
#                                calc_transmit_spectrum=self.calc_transmit_spectrum,
#                                calc_receive_spectrum=self.calc_receive_spectrum,
#                                calc_transmit_stft=self.calc_transmit_stft,
#                                calc_receive_stft=self.calc_receive_stft,
#                                spectrum_fft_size=self.spectrum_fft_size,
#                                confidence_margin=self.confidence_margin,
#                                confidence_level=self.confidence_level,
#                                confidence_metric=self.confidence_metric,
#                                min_num_drops=self.min_num_drops,
#                                max_num_drops=self.max_num_drops)
#
#        # Save most recent drop
#        drop: Optional[SimulationDrop] = None
#
#        for noise_index, snr in enumerate(self.noise_loop):
#
#            for d in range(self.max_num_drops):
#
#                drop_run_flag = statistics.run_flag_matrix[:, :, noise_index]
#
#                # Prematurely abort the drop loop if all stopping criteria have been met
#                if np.sum(drop_run_flag.flatten()) == 0:
#
#                    if self.verbosity.value <= Verbosity.INFO.value:
#
#                        info_str = f" Stopping criteria for SNR tap #{noise_index} met "
#                        padding = .5 * max(65 - len(info_str), 0)
#                        print('-' * floor(padding) + info_str + '-' * ceil(padding))
#
#                    break
#
#                drop = self.drop(snr, drop_run_flag)
#
#                # Print drop statistics if verbosity flag is set
#                if self.verbosity.value <= Verbosity.INFO.value:
#
#                    bers = drop.bit_error_rates
#                    blers = drop.block_error_rates
#
#                    for tx_id, (tx_bers, tx_blers) in enumerate(zip(bers, blers)):
#                        for rx_id, (ber, bler) in enumerate(zip(tx_bers, tx_blers)):
#
#                            link_str = f"{tx_id}x{rx_id}"
#                            ber_str = "-" if ber is None else f"{ber:.4f}"
#                            bler_str = "-" if bler is None else f"{bler:.4f}"
#
#                            if tx_id == 0 and rx_id == 0:
#
#                                snr_str = f"{10 * np.log10(snr):.1f}"
#                                print(f"{snr_str:<10}{d:<10}{link_str:<15}{ber_str:<15}{bler_str:<15}")
#
#                            else:
#                                print(" " * 20 + f"{link_str:<15}{ber_str:<15}{bler_str:<15}")
#
#                # Visualize plot if requested
#                if self.plot_drop:
#
#                    if self.plot_drop_transmitted_bits:
#                        drop.plot_transmitted_bits()
#
#                    if self.plot_drop_transmitted_signals:
#                        drop.plot_transmitted_signals()
#
#                    if self.plot_drop_received_signals:
#                        drop.plot_received_signals()
#
#                    if self.plot_drop_received_bits:
#                        drop.plot_received_bits()
#
#                    if self.plot_drop_bit_errors:
#                        drop.plot_bit_errors()
#
#                    if self.plot_drop_transmitted_symbols:
#                        drop.plot_transmitted_symbols()
#
#                    if self.plot_drop_received_symbols:
#                        drop.plot_received_symbols()
#
#                    if self.plot_drop_block_errors:
#                        drop.plot_block_errors()
#
#                    if self.plot_drop_transmit_stft:
#                        drop.plot_transmit_stft()
#
#                    if self.plot_drop_receive_stft:
#                        drop.plot_receive_stft()
#
#                    if self.plot_drop_transmit_spectrum:
#                        drop.plot_transmit_spectrum()
#
#                    if self.plot_drop_receive_spectrum:
#                        drop.plot_receive_spectrum()
#
#                    plt.show()
#
#                # Add drop to the statistics
#                statistics.add_drop(drop, noise_index)
#
#                # Check confidence if the routine is enabled and
#                # the minimum number of configured drops has been reached
#                # if self.confidence_metric is not ConfidenceMetric.DISABLED and d >= self.min_num_drops:
#
#        # Dump statistics results
#        # statistics.save(self.results_dir)
#
#        # Plot statistics results, if flags apply
#        if self.calc_transmit_spectrum:
#            statistics.plot_transmit_spectrum()
#
#        if self.calc_receive_spectrum:
#            statistics.plot_receive_spectrum()
#
#        if self.plot_bit_error:
#            statistics.plot_bit_error_rates()
#
#        if self.plot_block_error:
#            statistics.plot_block_error_rates()
#
#        # Plot the last drop to visualize stfts, if available
#        if drop is not None:
#
#            if self.calc_transmit_stft:
#                drop.plot_transmit_stft()
#
#            if self.calc_receive_stft:
#                drop.plot_receive_stft()
#
#        plt.show()
#        return statistics

    def drop(self,
             snr: float,
             drop_run_flag: Optional[np.ndarray] = None) -> SimulationDrop:
        """Generate a single simulation drop.

        Args:

            snr (float):
                The signal-to-noise-ratio at the receiver-side.

            drop_run_flag (np.ndarray, optional):
                Drop run flags.

        Returns:
            SimulationDrop: The generated drop.

        Raises:
            RuntimeError: If no scenario has been added to the simulation yet.
        """

        # Generate base-band signals, data symbols and data bits generated by each operator
        transmitted_signals, transmitted_symbols, transmitted_bits = self.transmit_operators(drop_run_flag=drop_run_flag)

        # Generate radio-frequency band signals emitted by each device
        transmitted_device_signals = self.transmit_devices()

        # Simulate propagation over channel model
        propagation_matrix = self.propagate(transmitted_device_signals, drop_run_flag)

        # Simulate signal reception and mixing at the receiver-side of devices
        received_device_signals = self.receive_devices(propagation_matrix, snr=snr)

        # Generate base-band signals, data symbols and data bits generated by each operator
        received_signals, received_symbols, received_bits = self.receive_operators(drop_run_flag)

        # Collect block sizes
        transmit_block_sizes = [1 for _ in self.transmitters]
        receive_block_sizes = [1 for _ in self.receivers]
        # ToDo: Re-implement block sizes
        # transmit_block_sizes = scenario.transmit_block_sizes
        # receive_block_sizes = scenario.receive_block_sizes

        # ToDo: Maybe change the structure here
        received_samples = [received_signals[0] for received_signal in received_signals]

        # Save generated signals
        drop = SimulationDrop(transmitted_bits, transmitted_symbols, transmitted_signals, transmit_block_sizes,
                              received_samples, received_symbols, received_bits, receive_block_sizes,
                              True, self.spectrum_fft_size)
        return drop

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
    def snr(self) -> Optional[float]:
        """Ratio of signal energy to noise power at the receiver-side.

        Returns:
            Optional[float]:
                Linear signal energy to noise power ratio.
                `None` if not specified.

        Raises:
            ValueError: On ratios smaller or equal to zero.
        """

        return self.__snr

    @snr.setter
    def snr(self, value: Optional[float]) -> None:
        """Set ratio of signal energy to noise power at the receiver-side"""

        if value is None:

            self.__snr = None

        else:

            if value <= 0.:
                raise ValueError("Signal to noise ratio must be greater than zero")

            self.__snr = value

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
            val (int): Minim number of drops.

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

    @property
    def drop_duration(self) -> float:
        """The scenario's default drop duration in seconds.

        If the drop duration is set to zero, the property will return the maximum frame duration
        over all registered transmitting modems as drop duration!

        Returns:
            float: The default drop duration in seconds.

        Raises:
            ValueError: For durations smaller than zero.
        """

        # Return the largest frame length as default drop duration
        if self.__drop_duration == 0.0:

            duration = 0.

            for device in self.devices:
                duration = max(duration, device.max_frame_duration)

            return duration

        else:
            return self.__drop_duration

    @drop_duration.setter
    def drop_duration(self, value: float) -> None:
        """Set the scenario's default drop duration."""

        if value < 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        self.__drop_duration = value

    def transmit_operators(self,
                           drop_run_flag: Optional[np.ndarray] = None,
                           drop_duration: Optional[float] = None) -> \
            Tuple[List[Optional[Signal]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """Generate base-band signal models emitted by all registered transmitting operators.

        Args:

            drop_run_flag (np.ndarray, optional):
                Mask that says if signals are to be created for specific snr.

            drop_duration (float, optional):
                Length of simulated transmission in seconds.

        Returns:
            (List[Optional[Signal]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]):

                baseband_signal (List[Optional[Signal]]):
                    List of baseband signals generated by each transmitting operator.

                data_symbols (List[Optional[np.ndarray]]):
                    List data symbols mapped by each transmitting operator.

                data_bits (List[Optional[np.ndarray]]):
                    List data bits transmitted by each transmitting operator.

        Raises:

            ValueError:
                On invalid `drop_duration`s.

            ValueError
                If `data_bits` does not contain data for each transmitting modem.
        """

        if drop_duration is None:
            drop_duration = self.drop_duration

        if drop_duration <= 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        if drop_run_flag is not None:
            sending_tx_idx = np.flatnonzero(np.sum(drop_run_flag, axis=1))
        else:
            sending_tx_idx = np.arange(len(self.transmitters))

        transmitted_signals: List[Optional[Signal]] = []
        transmitted_symbols: List[Optional[np.ndarray]] = []
        transmitted_bits: List[Optional[np.ndarray]] = []

        for transmitter_idx, transmitter in enumerate(self.transmitters):

            if transmitter_idx in sending_tx_idx:

                signal, symbols, bits = transmitter.transmit(drop_duration)
                transmitted_signals.append(signal)
                transmitted_symbols.append(symbols)
                transmitted_bits.append(bits)

            else:

                transmitted_signals.append(None)
                transmitted_symbols.append(None)
                transmitted_bits.append(None)

        return transmitted_signals, transmitted_symbols, transmitted_bits

    def transmit_devices(self) -> List[List[Signal]]:
        """Generate radio-frequency band signal models emitted by devices.

        Returns:
            List[List[Signal]]:
                List of signal models emitted by transmitting devices.
        """

        return [device.transmit() for device in self.devices]

    def propagate(self,
                  transmitted_signals: Optional[List[List[Optional[Signal]]]],
                  drop_run_flag: Optional[np.ndarray] = None) -> np.ndarray:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension

        Args:

            transmitted_signals (List[List[Optional[np.ndarray]]]):
                Signal models transmitted by each  registered device.

            drop_run_flag (np.ndarray, optional): Mask that says if signals are to be created for specific snr.

        Returns:

            np.ndarray:
                A square matrix of dimension `num_devices` containing tuples of propagated signals as well as the
                respective channel state information.

        Raises:

            ValueError:
                If the number of `transmitted_signals` does not equal the number of devices.
        """

        if transmitted_signals is None:
            transmitted_signals = [device.transmit() for device in self.devices]

        if len(transmitted_signals) != self.num_devices:
            raise ValueError(f"Number of transmit signals ({len(transmitted_signals)}) does not match "
                             f"the number of registered devices ({self.num_devices})")

        # Initialize the propagated signals
        propagation_matrix = np.empty((self.num_devices, self.num_devices), dtype=object)

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        for device_alpha_idx, device_alpha in enumerate(self.devices):
            for device_beta_idx, device_beta in enumerate(self.devices[:(1+device_alpha_idx)]):

                alpha_transmissions = transmitted_signals[device_alpha_idx]
                beta_transmissions = transmitted_signals[device_beta_idx]

                channel: Channel = self.__channels[device_alpha_idx, device_beta_idx]
                alpha_receptions, beta_receptions, csi = channel.propagate(alpha_transmissions, beta_transmissions)

                propagation_matrix[device_alpha_idx, device_beta_idx] = (alpha_receptions, csi)
                propagation_matrix[device_beta_idx, device_alpha_idx] = (beta_receptions, csi)

        return propagation_matrix

    def receive_devices(self,
                        propagation_matrix: np.ndarray) -> List[Signal]:
        """Generate base-band signal models received by devices.

        Args:

            propagation_matrix (np.ndarray):
                Matrix of signals and channel states impinging onto devices.

        Returns:
            received_signals(List[Signal]):
                Signals received by the devices.

        Raises:

            ValueError:
                If the length of `propagation_matrix` does not equal the number of devices within this scenario.
        """

        if len(propagation_matrix) != self.num_devices:
            raise ValueError(f"Number of arriving signals ({len(propagation_matrix)}) does not match "
                             f"the number of receiving devices ({self.num_devices})")

        received_signals: List[Signal] = []
        for device, impinging_signals in zip(self.devices, propagation_matrix):

            baseband_signal = device.receive(device_signals=impinging_signals, snr=self.snr)
            received_signals.append(baseband_signal)

        return received_signals

    def receive_operators(self,
                          drop_run_flag: Optional[np.ndarray] = None) -> \
            Tuple[List[Optional[Signal]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """Generate signals received by all receivers registered with this scenario.

        This routine superimposes propagated signals at the receiver-inputs,
        introduces additive thermal noise and mixes according to carrier frequency configurations.

        Args:

            propagation_matrix (List[List[Tuple[Signal, ChannelStateInformation]]]):
                MxN Matrix of pairs of received signals and impulse responses.
                The entry in the M-th row and N-th column contains the propagation data between
                the N-th transmitter and M-th receiver.

            drop_run_flag (np.ndarray, optional):
                Mask that says if signals are to be created for specific snr.


        Returns:
            List[Tuple[Signal, ChannelStateInformation, float]]:
                A list of M tuples containing the received noisy base-band signals, the channel impulse response
                as well as the noise variance for each receiving modem, respectively.

        Raises:
            ValueError:
                If the number of `arriving_signals` does not equal the number of registered receive modems.
        """

        signals: List[Optional[Signal]] = []
        symbols: List[Optional[np.ndarray]] = []
        bits: List[Optional[np.ndarray]] = []

        for receiver in self.receivers:

            # Capture received signals and channel state information
            received_signal, received_symbols, received_bits = receiver.receive()

            signals.append(received_signal)
            symbols.append(received_symbols)
            bits.append(received_bits)

        return signals, symbols, bits

    @staticmethod
    def calculate_noise_variance(receiver: Receiver, snr: float, snr_type: SNRType) -> float:
        """ToDo: Docstring"""

        if snr_type == SNRType.EBN0:
            return receiver.waveform_generator.bit_energy / snr

        if snr_type == SNRType.ESN0:
            return receiver.waveform_generator.symbol_energy / snr

        if snr_type == SNRType.CUSTOM:  # TODO: What is custom exactly supposed to do?
            return 1 / snr

    @staticmethod
    def detect(scenario: Scenario,
               received_signals: List[Tuple[Signal, ChannelStateInformation, float]],
               drop_run_flag: Optional[np.ndarray] = None) -> Tuple[List[Optional[np.ndarray]],
                                                                    List[Optional[np.ndarray]]]:
        """Detect bits from base-band signals.

        Calls the waveform-generator's receive-chain routines of each respective receiver.

        Args:
            scenario (Scenario):
                The scenario for which to simulate bit detection.

            received_signals (List[Tuple[Signal, ChannelStateInformation, float]]):
                A list of M tuples containing the received noisy base-band signals, the channel impulse response
                as well as the noise variance for each receiving modem, respectively.

            drop_run_flag (np.ndarray, optional): Mask that says if signals are to be created for specific snr.

        Returns:
            List[np.ndarray]:
                A list of M bit streams, where the m-th entry contains the bits detected by the m-th
                receiving modem within the m-th base-band signal.

        Raises:
            ValueError:
                If `received_signals` contains less entries than receivers registered in `scenario`.
        """

        if scenario.num_receivers != len(received_signals):
            raise ValueError("Less received signals than scenario receivers provided")

        if drop_run_flag is None:
            active_rx_idx = np.arange(len(scenario.receivers))
        else:
            active_rx_idx = np.flatnonzero(np.all(drop_run_flag, axis=0))

        received_bits: List[Optional[np.ndarray]] = []
        received_symbols: List[Optional[np.ndarray]] = []

        for receiver, (signal, channel, noise) in zip(scenario.receivers, received_signals):

            # Only demodulate active receiver signals
            if receiver.index in active_rx_idx and signal is not None:

                bits, symbols = receiver.demodulate(signal, channel, noise)
                received_bits.append(bits)
                received_symbols.append(symbols)

            else:

                received_bits.append(None)
                received_symbols.append(None)

        return received_bits, received_symbols

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
        devices: List[SimulatedDevice] = state.pop('Devices', [])
        operators: List[Tuple[Any, int, ...]] = state.pop('Operators', [])
        channels: List[Tuple[Channel, int, ...]] = state.pop('Channels', [])

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

        # Add devices to the simulation
        for device in devices:
            simulation.add_device(device)

        # Assign operators their respective devices
        for operator_tuple in operators:

            operator = operator_tuple[0]
            device_index = operator_tuple[1]

            operator.device = simulation.devices[device_index]

        # Assign channel models
        for channel_tuple in channels:

            channel = channel_tuple[0]
            output_device_idx = channel_tuple[1]
            input_device_idx = channel_tuple[2]

            simulation.set_channel(output_device_idx, input_device_idx, channel)

        # Return simulation instance recovered from the serialization
        return simulation


@remote(num_cpus=1)
class SimulationActor(MonteCarloActor[Simulation]):

    def sample(self, simulation: Simulation) -> Simulation:

        # Generate base-band signals, data symbols and data bits generated by each operator
        transmitted_signals, transmitted_symbols, transmitted_bits = simulation.transmit_operators()

        # Generate radio-frequency band signals emitted by each device
        transmitted_device_signals = simulation.transmit_devices()

        # Simulate propagation over channel model
        propagation_matrix = simulation.propagate(transmitted_device_signals)

        # Simulate signal reception and mixing at the receiver-side of devices
        received_device_signals = simulation.receive_devices(propagation_matrix)

        # Generate base-band signals, data symbols and data bits generated by each operator
        received_signals, received_symbols, received_bits = simulation.receive_operators()

        return simulation
