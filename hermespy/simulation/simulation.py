# -*- coding: utf-8 -*-
"""
==========
Simulation
==========
"""

from __future__ import annotations
from typing import Any, List, Type, Optional, Union, Tuple
from typing import Any, Dict, List, Type, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from os import path
from ray import remote
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from ..core.executable import Executable, Verbosity
from ..core.device import Operator
from ..core.drop import Drop
from ..channel import QuadrigaInterface, Channel, ChannelStateInformation
from ..core.factory import Serializable
from ..core.monte_carlo import Evaluator, MonteCarlo, MonteCarloActor, MonteCarloResult
from ..core.scenario import Scenario
from ..core.signal_model import Signal
from ..core.statistics import SNRType
from .simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
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
    """HermesPy simulation configuration."""

    yaml_tag = u'Simulation'
    """YAML serialization tag."""

    snr_type: SNRType
    """Global type of signal to noise ratio."""

    plot_results: bool
    """Plot results after simulation runs"""

    dump_results: bool
    """Dump results to files after simulation runs."""

    __channels: np.ndarray
    __operators: List[Operator]
    __snr: Optional[float]

    def __init__(self,
                 num_samples: int = 100,
                 drop_duration: float = 0.,
                 plot_results: bool = False,
                 dump_results: bool = True,
                 snr_type: Union[str, SNRType] = SNRType.EBN0,
                 results_dir: Optional[str] = None,
                 verbosity: Union[str, Verbosity] = Verbosity.INFO,
                 seed: Optional[int] = None) -> None:
        """Args:

            drop_duration(float, optional):
                Duration of simulation drops in seconds.

            plot_results (bool, optional):
                Plot results after simulation runs.
                Disabled by default.

            dump_results (bool, optional):
                Dump results to files after simulation runs.
                Enabled by default.

            snr_type (Union[str, SNRType]):
                The signal to noise ratio metric to be used.

            results_dir (str, optional):
                Directory in which all simulation artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        # Initialize base classes
        Executable.__init__(self, results_dir, verbosity)
        Scenario.__init__(self, seed=seed)
        MonteCarlo.__init__(self, investigated_object=self, num_samples=num_samples)

        self.__channels = np.ndarray((0, 0), dtype=object)
        self.plot_results = plot_results
        self.dump_results = dump_results
        self.drop_duration = drop_duration
        self.snr_type = snr_type
        self.snr = None
        self.__operators: List[Operator] = []

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

    def set_channel(self,
                    receiver: Union[int, SimulatedDevice],
                    transmitter: Union[int, SimulatedDevice],
                    channel: Channel) -> None:
        """Specify a channel within the channel matrix.

        Args:

            receiver (int):
                Index of the receiver within the channel matrix.

            transmitter (int):
                Index of the transmitter within the channel matrix.

            channel (Channel):
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).

        Raises:
            ValueError:
                If `transmitter_index` or `receiver_index` are greater than the channel matrix dimensions.
        """

        if isinstance(receiver, SimulatedDevice):
            receiver = self.devices.index(receiver)

        if isinstance(transmitter, SimulatedDevice):
            transmitter = self.devices.index(transmitter)

        if self.__channels.shape[0] <= transmitter or 0 > transmitter:
            raise ValueError("Transmitter index greater than channel matrix dimension")

        if self.__channels.shape[1] <= receiver or 0 > receiver:
            raise ValueError("Receiver index greater than channel matrix dimension")

        # Update channel field within the matrix
        self.__channels[transmitter, receiver] = channel

        # Set proper receiver and transmitter fields
        channel.transmitter = self.devices[transmitter]
        channel.receiver = self.devices[receiver]
        channel.random_mother = self
        channel.scenario = self

    def run(self) -> MonteCarloResult[Scenario[SimulatedDevice]]:

        # Generate simulation result
        result = self.simulate(SimulationActor)

        # Visualize results if the flag is enabled
        if self.plot_results:

            with self.style_context():

                result.plot()
                plt.show()

        if self.dump_results:
            result.save_to_matlab(path.join(self.results_dir, 'results.mat'))

        return result

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

        state = constructor.construct_mapping(node, deep=True)

        # Launch a global quadriga instance
        quadriga_interface: Optional[QuadrigaInterface] = state.pop(QuadrigaInterface.yaml_tag, None)
        if quadriga_interface is not None:
            QuadrigaInterface.SetGlobalInstance(quadriga_interface)

        # Pop configuration sections for "special" treatment
        devices: List[SimulatedDevice] = state.pop('Devices', [])
        channels: List[Tuple[Channel, int, ...]] = state.pop('Channels', [])
        operators: List[Operator] = state.pop('Operators', [])
        evaluators: List[Evaluator] = state.pop('Evaluators', [])
        dimensions: Dict[str, Any] = state.pop('Dimensions', {})

        # Initialize simulation
        simulation = cls.InitializationWrapper(state)

        # Add devices to the simulation
        for device in devices:
            simulation.add_device(device)

        # Assign channel models
        for channel, channel_position in channels:

            output_device_idx = channel_position[0]
            input_device_idx = channel_position[1]

            simulation.set_channel(output_device_idx, input_device_idx, channel)

        # Register operators
        for operator in operators:
            simulation.__operators.append(operator)

        # Register evaluators
        for evaluator in evaluators:
            simulation.add_evaluator(evaluator)

        # Add simulation dimensions
        for dimension_key, dimension_values in dimensions.items():
            simulation.add_dimension(dimension_key, dimension_values)

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
