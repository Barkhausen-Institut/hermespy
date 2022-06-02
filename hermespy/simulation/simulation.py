# -*- coding: utf-8 -*-
"""
==========
Simulation
==========
"""

from __future__ import annotations
from typing import Any, Dict, List, Type, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from os import path
from ray import remote
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from ..core.executable import Executable, Verbosity
from ..core.device import Operator
from ..channel import QuadrigaInterface, Channel
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
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulationScenario(Scenario[SimulatedDevice]):

    __channels: np.ndarray      # Channel matrix linking devices
    __snr: Optional[float]      # Signal to noise ratio at the receiver-side

    def __init__(self,
                 seed: Optional[int] = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        Scenario.__init__(self, seed=seed)
        self.__channels = np.ndarray((0, 0), dtype=object)

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

        if self.num_devices == 1:

            self.__channels = np.array([[Channel(device, device)]], dtype=object)

        else:

            # Create new channels from each existing device to the newly added device
            new_channels = np.array([[Channel(device, rx)] for rx in self.devices])

            # Complete channel matrix by the newly created channels
            self.__channels = np.append(self.__channels, new_channels[:-1], axis=1)
            self.__channels = np.append(self.__channels, new_channels.T, axis=0)

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
        channels: List[Channel] = self.__channels[:, transmitter_index].tolist()

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
        channels: List[Channel] = self.__channels[receiver_index, ].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def set_channel(self,
                    receiver: Union[int, SimulatedDevice],
                    transmitter: Union[int, SimulatedDevice],
                    channel: Optional[Channel]) -> None:
        """Specify a channel within the channel matrix.

        Args:

            receiver (int):
                Index of the receiver within the channel matrix.

            transmitter (int):
                Index of the transmitter within the channel matrix.

            channel (Optional[Channel]):
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
        self.__channels[receiver, transmitter] = channel

        if channel is not None:

            # Set proper receiver and transmitter fields
            channel.transmitter = self.devices[transmitter]
            channel.receiver = self.devices[receiver]
            channel.random_mother = self
            channel.scenario = self

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


SimulationResult: Type[MonteCarloResult[SimulationScenario]]
"""Result Container of a Full Simulation Campaign."""


class SimulationRunner(object):

    __scenario: SimulationScenario      # Scenario to be run

    def __init__(self,
                 scenario: SimulationScenario) -> None:
        """
        Args:

            scenario(SimulationScenario):
                Scenario to be run.
        """

        self.__scenario = scenario

    def transmit_operators(self,
                           drop_duration: Optional[float] = None) -> None:
        """Generate base-band signal models emitted by all registered transmitting operators.

        Args:

            drop_duration (float, optional):
                Length of simulated transmission in seconds.

        Raises:

            ValueError:
                On invalid `drop_duration`s.

            ValueError
                If `data_bits` does not contain data for each transmitting modem.
        """

        # Infer drop duration from scenario if not provided
        drop_duration = self.__scenario.drop_duration if drop_duration is None else drop_duration

        if drop_duration <= 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        for transmitter_idx, transmitter in enumerate(self.__scenario.transmitters):
            transmitter.transmit()

    def transmit_devices(self) -> List[List[Signal]]:
        """Generate radio-frequency band signal models emitted by devices.

        Returns:
            List[List[Signal]]:
                List of signal models emitted by transmitting devices.
        """

        return [device.transmit() for device in self.__scenario.devices]

    def propagate(self,
                  transmitted_signals: Optional[List[List[Optional[Signal]]]]) -> np.ndarray:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension

        Args:

            transmitted_signals (List[List[Optional[np.ndarray]]]):
                Signal models transmitted by each  registered device.

        Returns:

            np.ndarray:
                A square matrix of dimension `num_devices` containing tuples of propagated signals as well as the
                respective channel state information.

        Raises:

            ValueError:
                If the number of `transmitted_signals` does not equal the number of devices.
        """

        if transmitted_signals is None:
            transmitted_signals = [device.transmit() for device in self.__scenario.devices]

        if len(transmitted_signals) != self.__scenario.num_devices:
            raise ValueError(f"Number of transmit signals ({len(transmitted_signals)}) does not match "
                             f"the number of registered devices ({self.__scenario.num_devices})")

        # Initialize the propagated signals
        propagation_matrix = np.empty((self.__scenario.num_devices, self.__scenario.num_devices), dtype=object)

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        for device_alpha_idx, device_alpha in enumerate(self.__scenario.devices):
            for device_beta_idx, device_beta in enumerate(self.__scenario.devices[:(1+device_alpha_idx)]):

                alpha_transmissions = transmitted_signals[device_alpha_idx]
                beta_transmissions = transmitted_signals[device_beta_idx]

                channel: Channel = self.__scenario.channels[device_alpha_idx, device_beta_idx]
                beta_receptions, alpha_receptions, csi = channel.propagate(alpha_transmissions, beta_transmissions)

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

        if len(propagation_matrix) != self.__scenario.num_devices:
            raise ValueError(f"Number of arriving signals ({len(propagation_matrix)}) does not match "
                             f"the number of receiving devices ({self.__scenario.num_devices})")

        received_signals: List[Signal] = []
        for device, impinging_signals in zip(self.__scenario.devices, propagation_matrix):

            baseband_signal = device.receive(device_signals=impinging_signals, snr=self.__scenario.snr)
            received_signals.append(baseband_signal)

        return received_signals


@remote(num_cpus=1)
class SimulationActor(MonteCarloActor[SimulationScenario], SimulationRunner):

    def __init__(self, argument_tuple: Any) -> None:
        """
        Args:

            argument_tuple (Any):
                MonteCarloActor initialization arguments.
        """

        MonteCarloActor.__init__(self, argument_tuple)
        SimulationRunner.__init__(self, self._investigated_object)

    def sample(self) -> SimulationScenario:

        # Generate base-band signals, data symbols and data bits generated by each operator
        self.transmit_operators()

        # Generate radio-frequency band signals emitted by each device
        transmitted_device_signals = self.transmit_devices()

        # Simulate propagation over channel model
        propagation_matrix = self.propagate(transmitted_device_signals)

        # Simulate signal reception and mixing at the receiver-side of devices
        received_device_signals = self.receive_devices(propagation_matrix)

        # Generate base-band signals, data symbols and data bits generated by each operator
        [receiver.receive() for receiver in self._investigated_object.receivers]

        return self._investigated_object


class Simulation(Executable, Serializable, MonteCarlo[SimulationScenario]):
    """HermesPy simulation configuration."""

    yaml_tag = u'Simulation'
    """YAML serialization tag."""

    snr_type: SNRType
    """Global type of signal to noise ratio."""

    plot_results: bool
    """Plot results after simulation runs"""

    dump_results: bool
    """Dump results to files after simulation runs."""

    __scenario: SimulationScenario
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

        Executable.__init__(self, results_dir, verbosity)

        self.__scenario = SimulationScenario(seed=seed)
        self.plot_results = plot_results
        self.dump_results = dump_results
        self.drop_duration = drop_duration
        self.snr_type = snr_type
        self.snr = None
        self.__operators: List[Operator] = []

        MonteCarlo.__init__(self, investigated_object=self.__scenario, num_samples=num_samples, console=self.console)

    @property
    def scenario(self) -> SimulationScenario:
        """Scenario description of the simulation.

        Returns:

            SimulationScenario:
                Scenario description.
        """

        return self.__scenario

    def run(self) -> SimulationResult:

        # Print indicator that the simulation is starting
        self.console.print()  # Just an empty line
        self.console.rule("Simulation Campaign")
        self.console.print()  # Just an empty line

        # Generate simulation result
        result = self.simulate(SimulationActor)

        # Visualize results if the flag respective is enabled
        figures: List[plt.Figure] = []
        if self.plot_results:

            with self.style_context():
                figures = result.plot()

        # Dump results if the respective flag is enabled
        if self.dump_results and self.results_dir is not None:

            # Save figures to png files
            for figure_idx, figure in enumerate(figures):
                figure.savefig(path.join(self.results_dir, f'figure_{figure_idx}.png'), format='png')

            # Save results to matlab file
            result.save_to_matlab(path.join(self.results_dir, 'results.mat'))

        # Show plots if the flag is enabled
        if self.plot_results:
            plt.show()

        # Return result object
        return result

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

        return self.scenario.snr

    @snr.setter
    def snr(self, value: Optional[float]) -> None:
        self.scenario.snr = value

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
            "snr_type": node.snr_type.value,
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
        channels: List[Tuple[Channel, int]] = state.pop('Channels', [])
        operators: List[Operator] = state.pop('Operators', [])
        evaluators: List[Evaluator] = state.pop('Evaluators', [])
        dimensions: Dict[str, Any] = state.pop('Dimensions', {})

        # Initialize simulation
        simulation = cls.InitializationWrapper(state)

        # Add devices to the simulation
        for device in devices:
            simulation.scenario.add_device(device)

        # Assign channel models
        for channel, channel_position in channels:

            output_device_idx = channel_position[0]
            input_device_idx = channel_position[1]

            simulation.scenario.set_channel(output_device_idx, input_device_idx, channel)

        # Register operators
        for operator in operators:
            simulation.__operators.append(operator)

        # Register evaluators
        for evaluator in evaluators:
            simulation.add_evaluator(evaluator)

        # Add simulation dimensions
        for dimension_key, dimension_values in dimensions.items():
            simulation.new_dimension(dimension_key, dimension_values)

        # Return simulation instance recovered from the serialization
        return simulation
