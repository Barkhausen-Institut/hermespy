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

from hermespy.core import ConsoleMode, MonteCarloResult, SignalTransmitter, SignalReceiver, Signal
from hermespy.modem import DuplexModem, BitErrorEvaluator, RRCWaveform
from hermespy.simulation import N0
from hermespy.simulation.simulation import SimulatedDevice, Simulation, SimulationActor, SimulationRunner, SimulationScenario
from unit_tests.utils import random_rf_signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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
        self.oversampling_factor = 4
        self.bandwidth = self.sampling_rate / self.oversampling_factor
        self.num_samples = 100

        self.device_alpha = self.scenario.new_device(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor)
        self.device_beta = self.scenario.new_device(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor)

        transmitted_signal = random_rf_signal(1, self.num_samples, self.bandwidth, self.oversampling_factor)
        self.transmitter_alpha = SignalTransmitter(transmitted_signal)
        self.transmitter_beta = SignalTransmitter(transmitted_signal)
        self.receiver_alpha = SignalReceiver(self.num_samples)
        self.receiver_beta = SignalReceiver(self.num_samples)

        self.device_alpha.transmitters.add(self.transmitter_alpha)
        self.device_beta.transmitters.add(self.transmitter_beta)
        self.device_alpha.receivers.add(self.receiver_alpha)
        self.device_beta.receivers.add(self.receiver_beta)

        self.runner = SimulationRunner(self.scenario)

    def test_stages(self) -> None:
        """Make sure the stages all execute without exceptions"""

        # Realize channels
        self.runner.realize_channels()

        # Sample trajectories
        self.runner.sample_states()

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

    def test_propagate_validation(self) -> None:
        """Propagate should raise RuntimeErrors for invalid internal states"""

        with self.assertRaises(RuntimeError):
            self.runner.propagate()

    def test_process_inputs_validation(self) -> None:
        """Process inputs should raise RuntimeErrors for invalid internal states"""

        self.runner.realize_channels()
        self.runner.sample_states()

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

        # Invalid number of impinging signals
        self.runner.sample_states()
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
        self.actor = SimulationActor(Mock(), (self.scenario, self.dimensions, []), 0)


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
        self.modem.waveform = RRCWaveform(num_preamble_symbols=0, num_data_symbols=10)
        self.device.transmitters.add(self.modem)
        self.device.receivers.add(self.modem)

        self.evaluator = BitErrorEvaluator(self.modem, self.modem)

        self.dimension = self.simulation.new_dimension("noise_level", [N0(p) for p in (1, 2, 3)])
        self.dimension = self.simulation.new_dimension("carrier_frequency", [0, 1e9], self.device)
        self.simulation.add_evaluator(self.evaluator)

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

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
        
    def test_devices(self) -> None:
        """Devices property should return the devices in the simulation"""
        
        new_device = self.simulation.new_device()
        self.assertIn(new_device, self.simulation.devices)
        
    def test_drop(self) -> None:
        """Test the drop generation routine"""
        
        drop = self.simulation.drop(0.1234)
        self.assertEqual(1, drop.num_device_receptions)
        self.assertEqual(1, drop.num_device_transmissions)

    def test_run(self) -> None:
        """Test running the simulation"""

        self.simulation.plot_results = True
        self.simulation.dump_results = True
        mock_visualization = Mock()
        mock_visualization.figure = Mock()

        with patch("hermespy.core.pymonte.monte_carlo.MonteCarloResult.plot", return_value=[mock_visualization]), TemporaryDirectory() as temp:
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

    def test_pip_pacakges(self) -> None:
        """Test the pip packages property"""

        self.assertCountEqual(["ray", "numpy", "scipy", "matplotlib", "rich", "sparse", "protobuf", "numba"], self.simulation._pip_packages())
