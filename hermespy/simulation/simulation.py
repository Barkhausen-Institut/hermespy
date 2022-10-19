# -*- coding: utf-8 -*-
"""
==========
Simulation
==========
"""

from __future__ import annotations
from re import S
from typing import Any, Callable, Dict, List, Type, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from os import path
from ray import remote
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from hermespy.core import Executable, Verbosity, Operator, Serializable, ConsoleMode, Evaluator, dimension, \
    MonteCarloActor, MonteCarlo, MonteCarloResult, Scenario, Signal, SNRType
from hermespy.channel import QuadrigaInterface, Channel
from .simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"
    

class SimulationScenario(Scenario[SimulatedDevice]):

    __channels: np.ndarray      # Channel matrix linking devices
    __snr: Optional[float]      # Signal to noise ratio at the receiver-side
    __snr_type: SNRType         # Global global type of signal to noise ratio.

    def __init__(self,
                 seed: Optional[int] = None,
                 snr: float = float('inf'),
                 snr_type: Union[str, SNRType] = SNRType.PN0) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
                
            snr (float, optional):
                The assumed linear signal to noise ratio.
                Infinite by default, i.e. no added noise during reception.

            snr_type (Union[str, SNRType], optional):
                The signal to noise ratio metric to be used.
                By default, signal power to noise power is assumed.
        """

        Scenario.__init__(self, seed=seed)
        self.snr = snr
        self.snr_type = snr_type
        self.__channels = np.ndarray((0, 0), dtype=object)

    def new_device(self, *args, **kwargs) -> SimulatedDevice:
        """Add a new device to the simulation scenario.

        Returns:
            SimulatedDevice: Newly added simulated device.
        """

        device = SimulatedDevice(*args, **kwargs)
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

    @dimension
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

    @snr.setter(first_impact='receive_devices')
    def snr(self, value: Optional[float]) -> None:
        """Set ratio of signal energy to noise power at the receiver-side"""

        if value is None:

            self.__snr = None

        else:

            if value <= 0.:
                raise ValueError("Signal to noise ratio must be greater than zero")

            self.__snr = value
            
    @dimension
    def snr_type(self) -> SNRType:
        """Type of signal-to-noise ratio.

        Returns:
            SNRType: The SNR type.
        """

        return self.__snr_type

    @snr_type.setter(first_impact='receive_devices')
    def snr_type(self, snr_type: Union[str, int, SNRType]) -> None:
        """Modify the type of signal-to-noise ratio.

        Args:
            snr_type (Union[str, int, SNRType]):
                The new type of signal to noise ratio, string or enum representation.
        """

        if isinstance(snr_type, str):
            snr_type = SNRType[snr_type]
            
        if isinstance(snr_type, int):
            snr_type = SNRType(int)

        self.__snr_type = snr_type


class SimulationRunner(object):

    __scenario: SimulationScenario      # Scenario to be run
    __recent_device_transmissions: Optional[List[Signal]]
    __recent_propagation: Optional[np.ndarray]

    def __init__(self,
                 scenario: SimulationScenario) -> None:
        """
        Args:

            scenario(SimulationScenario):
                Scenario to be run.
        """

        self.__scenario = scenario
        self.__recent_device_transmissions = None
        self.__recent_propagation = None

    def transmit_operators(self) -> None:
        """Generate base-band signal models emitted by all registered transmitting operators."""

        for transmitter in self.__scenario.transmitters:
            _ = transmitter.transmit()

    def transmit_devices(self) -> None:
        """Generate radio-frequency band signal models emitted by devices"""

        device_transmssions = [device.transmit(clear_cache=False) for device in self.__scenario.devices]
        self.__recent_device_transmissions = device_transmssions
            
    def propagate(self) -> None:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension.
        """

        transmitted_signals = self.__recent_device_transmissions
        
        if transmitted_signals is None:
            raise RuntimeError("Propagation simulation stage called without prior device transmission")

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

                propagation_matrix[device_alpha_idx, device_beta_idx] = (alpha_receptions, csi.reciprocal())
                propagation_matrix[device_beta_idx, device_alpha_idx] = (beta_receptions, csi)

        # Cache the recent propagation
        self.__recent_propagation = propagation_matrix

    def receive_devices(self) -> None:
        """Generate base-band signal models received by devices."""
        
        propagation_matrix = self.__recent_propagation
        
        if propagation_matrix is None:
            raise RuntimeError("Receive device simulation stage called without prior channel propagation")

        if len(propagation_matrix) != self.__scenario.num_devices:
            raise ValueError(f"Number of arriving signals ({len(propagation_matrix)}) does not match "
                             f"the number of receiving devices ({self.__scenario.num_devices})")

        for device, impinging_signals in zip(self.__scenario.devices, propagation_matrix):
            _ = device.receive(device_signals=impinging_signals,
                               snr=self.__scenario.snr, snr_type=self.__scenario.snr_type)

    def receive_operators(self) -> None:
        """Demodulate base-band signal models received by all registered receiving operators."""

        for operator in self.__scenario.receivers:
            _ = operator.receive()

@remote(num_cpus=1)
class SimulationActor(MonteCarloActor[SimulationScenario], SimulationRunner):

    def __init__(self,
                 argument_tuple: Any,
                 index: int,
                 catch_exceptions: bool = True) -> None:
        """
        Args:

            argument_tuple (Any):
                MonteCarloActor initialization arguments.
        """

        MonteCarloActor.__init__(self, argument_tuple, index, catch_exceptions)
        SimulationRunner.__init__(self, self._investigated_object)

        # Update the internal random seed pseudo-deterministically for each actor instance
        seed = self._investigated_object._rng.integers(0, 100000000000000)
        individual_seed = seed + index * 12345678
        self._investigated_object.set_seed(individual_seed)
        
    @staticmethod
    def stage_identifiers() -> List[str]:
        return ['transmit_operators', 'transmit_devices', 'propagate', 'receive_devices', 'receive_operators']
    
    def stage_executors(self) -> List[Callable]:
        return [self.transmit_operators, self.transmit_devices, self.propagate, self.receive_devices, self.receive_operators]


class Simulation(Executable, Serializable, MonteCarlo[SimulationScenario]):
    """HermesPy simulation configuration."""

    yaml_tag = u'Simulation'
    """YAML serialization tag."""

    plot_results: bool
    """Plot results after simulation runs"""

    dump_results: bool
    """Dump results to files after simulation runs."""

    __scenario: SimulationScenario

    def __init__(self,
                 num_samples: int = 100,
                 drop_duration: float = 0.,
                 plot_results: bool = False,
                 dump_results: bool = True,
                 console_mode: ConsoleMode = ConsoleMode.INTERACTIVE,
                 ray_address: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 verbosity: Union[str, Verbosity] = Verbosity.INFO,
                 seed: Optional[int] = None,
                 num_actors: Optional[int] = None,
                 *args, **kwargs) -> None:
        """Args:

            drop_duration(float, optional):
                Duration of simulation drops in seconds.

            plot_results (bool, optional):
                Plot results after simulation runs.
                Disabled by default.

            dump_results (bool, optional):
                Dump results to files after simulation runs.
                Enabled by default.

            ray_address (str, optional):
                The address of the ray head node.
                If None is provided, the head node will be launched in this machine.

            results_dir (str, optional):
                Directory in which all simulation artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        Executable.__init__(self, results_dir, verbosity)

        self.__scenario = SimulationScenario(seed=seed, *args, **kwargs)
        self.plot_results = plot_results
        self.dump_results = dump_results
        self.drop_duration = drop_duration

        MonteCarlo.__init__(self, self.__scenario, num_samples,
                            console=self.console, console_mode=console_mode, ray_address=ray_address,
                            num_actors=num_actors)
        return

    @property
    def scenario(self) -> SimulationScenario:
        """Scenario description of the simulation.

        Returns:

            SimulationScenario:
                Scenario description.
        """

        return self.__scenario

    def run(self) -> MonteCarloResult[SimulationScenario]:

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
        # if self.plot_results:
        #    plt.show()

        # Return result object
        return result

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
        channels: List[Channel] = state.pop('Channels', [])
        operators: List[Operator] = state.pop('Operators', [])
        evaluators: List[Evaluator] = state.pop('Evaluators', [])
        dimensions: Dict[str, Any] = state.pop('Dimensions', {})

        # Initialize simulation
        simulation: Simulation = cls.InitializationWrapper(state)

        # Add devices to the simulation
        for device in devices:
            simulation.scenario.add_device(device)

        # Assign channel models
        for channel in channels:

            # If the scenario features just a single device, we can infer the transmitter and receiver easily
            if channel.transmitter is None or channel.receiver is None:
                
                if simulation.scenario.num_devices > 1:
                    raise RuntimeError("Please specifiy the transmitting and receiving device of each channel in a multi-device scenario")

                channel.transmitter = simulation.scenario.devices[0]
                channel.receiver = simulation.scenario.devices[0]

            simulation.scenario.set_channel(channel.receiver, channel.transmitter, channel)

        # Register evaluators
        for evaluator in evaluators:
            simulation.add_evaluator(evaluator)

        # Add simulation dimensions
        for dimension_key, dimension_values in dimensions.items():
            simulation.new_dimension(dimension_key, dimension_values)

        # Return simulation instance recovered from the serialization
        return simulation

    @staticmethod
    def _pip_packages() -> List[str]:
        
        return MonteCarlo._pip_packages() + [
            'sparse', 'protobuf', 'numba', 
        ]
