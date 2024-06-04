# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
from io import StringIO
from os import getenv
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import ray
from rich.console import Console

from hermespy.channel import IdealChannel
from hermespy.core import ConsoleMode, Factory, MonteCarloResult, SignalTransmitter, SignalReceiver, Signal
from hermespy.modem import DuplexModem, BitErrorEvaluator, RRCWaveform
from hermespy.simulation import N0
from hermespy.simulation.simulation import SimulatedDevice, Simulation, SimulationActor, SimulationRunner, SimulationScenario
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


GENERATE_OUTPUT = getenv("HERMES_TEST_PLOT", "False").lower() == "true"


class TestSimulationRunner(TestCase):
    """Test the Simulation Runner"""

    def setUp(self) -> None:
        self.seed = 0
        self.scenario = SimulationScenario(seed=self.seed)

        self.sampling_rate = 1e8
        self.num_samples = 100

        self.device_alpha = self.scenario.new_device(sampling_rate=self.sampling_rate)
        self.device_beta = self.scenario.new_device(sampling_rate=self.sampling_rate)

        transmitted_signal = Signal.Create(self.scenario._rng.standard_normal((1, self.num_samples)), self.sampling_rate)
        self.transmitter_alpha = SignalTransmitter(transmitted_signal)
        self.transmitter_beta = SignalTransmitter(transmitted_signal)
        self.receiver_alpha = SignalReceiver(self.num_samples, self.sampling_rate)
        self.receiver_beta = SignalReceiver(self.num_samples, self.sampling_rate)

        self.device_alpha.transmitters.add(self.transmitter_alpha)
        self.device_beta.transmitters.add(self.transmitter_beta)
        self.device_alpha.receivers.add(self.receiver_alpha)
        self.device_beta.receivers.add(self.receiver_beta)

        self.runner = SimulationRunner(self.scenario)

    def test_stages(self) -> None:
        """Make sure the stages all execute without exceptions"""

        try:
            
            # Realize channels
            self.runner.realize_channels()
            
            # Sample trajectories
            self.runner.sample_trajectories()
            
            # Transmit operators
            self.runner.transmit_operators()

            # Generate device outputs
            self.runner.generate_outputs()

            # Propagate device outputs
            self.runner.propagate()

            # Process device inputs
            self.runner.process_inputs()

            # Receive operators
            self.runner.receive_operators()

        except Exception as e:
            self.fail(str(e))

    def test_propagate_validation(self) -> None:
        """Propagate should raise RuntimeErrors for invalid internal states"""

        with self.assertRaises(RuntimeError):
            self.runner.propagate()

    def test_process_inputs_validation(self) -> None:
        """Process inputs should raise RuntimeErrors for invalid internal states"""

        self.runner.realize_channels()
        self.runner.sample_trajectories()

        # No trigger realizations cached
        with self.assertRaises(RuntimeError):
            self.runner.process_inputs()
            
        self.runner.transmit_operators()
        self.runner.generate_outputs()
        
        # No propagation matrix cached
        with self.assertRaises(RuntimeError):
            self.runner.process_inputs()
            
        self.runner.propagate()

        # Invalid number of trigger realizations
        _ = self.scenario.new_device()        
        with self.assertRaises(RuntimeError):
            self.runner.process_inputs()
            
        # Invalid number of impinging signals+
        self.runner.transmit_operators()
        self.runner.generate_outputs()
        with self.assertRaises(RuntimeError):
            self.runner.process_inputs()

class TestSimulationActor(TestCase):
    """Test the Simulation Actor"""

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Shut down ray
        ray.shutdown()

    def setUp(self) -> None:
        self.seed = 0
        self.device_alpha = SimulatedDevice()
        self.device_beta = SimulatedDevice()

        self.scenario = SimulationScenario(seed=self.seed)
        self.scenario.add_device(self.device_alpha)
        self.scenario.add_device(self.device_beta)

        self.dimensions = []
        self.actor = SimulationActor.remote((self.scenario, self.dimensions, []), 0)

    def test_run(self) -> None:
        """Test running the simulation actor"""

        run_result = ray.get(self.actor.run.remote([tuple()]))
        self.assertEqual(1, len(run_result.samples))


class TestSimulation(TestCase):
    """Test the simulation executable, the base class for simulation operations"""

    def setUp(self) -> None:
        self.io = StringIO()
        self.console = Console(file=None if GENERATE_OUTPUT else self.io)
        self.simulation = Simulation(seed=42)
        self.simulation.console = self.console
        self.simulation.num_drops = 1

        self.device = self.simulation.new_device()
        self.modem = DuplexModem()
        self.modem.waveform = RRCWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)
        self.modem.device = self.device
        self.evaluator = BitErrorEvaluator(self.modem, self.modem)

        self.dimension = self.simulation.new_dimension("noise_level", [N0(p) for p in (1, 2, 3)])
        self.dimension = self.simulation.new_dimension("carrier_frequency", [0, 1e9], self.device)
        self.simulation.add_evaluator(self.evaluator)

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Shut down ray
        ray.shutdown()

    def test_num_samples_setget(self) -> None:
        """Num samples property getter should return setter argument"""

        num_samples = 100
        self.simulation.num_samples = num_samples

        self.assertEqual(num_samples, self.simulation.num_samples)
        self.assertEqual(num_samples, self.simulation.num_drops)

    def test_console_mode_setget(self) -> None:
        """Console mode property getter should return setter argument"""

        console_mode = ConsoleMode.SILENT
        self.simulation.console_mode = console_mode

        self.assertEqual(console_mode, self.simulation.console_mode)

    def test_console_setget(self) -> None:
        """Console property getter should return setter argument"""

        console = Mock()
        self.simulation.console = console

        self.assertIs(console, self.simulation.console)

    def test_run(self) -> None:
        """Test running the simulation"""

        self.simulation.plot_results = True
        self.simulation.dump_results = True
        mock_visualization = Mock()
        mock_visualization.figure = Mock()

        with patch("hermespy.core.monte_carlo.MonteCarloResult.plot", return_value=[mock_visualization]), TemporaryDirectory() as temp:
            self.simulation.results_dir = temp
            result = self.simulation.run()

            self.assertIsInstance(result, MonteCarloResult)
            mock_visualization.figure.savefig.assert_called()

    def test_silent_run(self) -> None:
        """Test running the simulation without output"""

        self.simulation.console_mode = ConsoleMode.SILENT
        _ = self.simulation.run()

        if not GENERATE_OUTPUT:
            self.assertEqual("", self.io.getvalue())

    def test_set_channel(self) -> None:
        """Test the channel set convenience method"""

        expected_channel = Mock()
        self.simulation.set_channel(self.device, self.device, expected_channel)

        self.assertIs(expected_channel, self.simulation.scenario.channel(self.device, self.device))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.simulation)

    def test_serialization_channel_device_inference(self) -> None:
        """Test YAML serialization with channel device inference"""

        serialization = """
        !<Simulation>
            Devices:
                - &device !<SimulatedDevice>

            Channels:
                - [ *device, *device, !<Channel> ]
        """

        factory = Factory()
        simulation: Simulation = factory.from_str(serialization)

        self.assertEqual(1, len(simulation.scenario.devices))
        device = simulation.scenario.devices[0]
        channel = simulation.scenario.channel(device, device)
        self.assertIsInstance(channel, IdealChannel)

    def test_serialization_dimension_shorthand(self) -> None:
        """Test YAML serialization with dimension shorthand"""

        serialization = """
        !<Simulation>
            Devices:
                - !<SimulatedDevice>

            Dimensions:
                noise_level: [1, 2, 3]
        """

        factory = Factory()
        simulation = factory.from_str(serialization)

        self.assertEqual(1, len(simulation.dimensions))
        self.assertEqual("noise_level", simulation.dimensions[0].dimension)
        self.assertSequenceEqual([1, 2, 3], [p.value for p in simulation.dimensions[0].sample_points])

    def test_pip_pacakges(self) -> None:
        """Test the pip packages property"""

        self.assertCountEqual(["ray", "numpy", "scipy", "matplotlib", "rich", "sparse", "protobuf", "numba"], self.simulation._pip_packages())
