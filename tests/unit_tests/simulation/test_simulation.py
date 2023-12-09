# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable"""

from __future__ import annotations
import logging
from io import StringIO
from os import getenv
from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import ray
from h5py import File
from rich.console import Console

from hermespy.core import ConsoleMode, Factory, MonteCarloResult, SignalTransmitter, SignalReceiver, Signal
from hermespy.modem import DuplexModem, BitErrorEvaluator, RRCWaveform
from hermespy.simulation import StaticTrigger
from hermespy.simulation.simulation import SimulatedDevice, SimulatedDrop, Simulation, SimulationActor, SimulationRunner, SimulationScenario, SNRType
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


GENERATE_OUTPUT = getenv("HERMES_TEST_PLOT", "False").lower() == "true"


class TestSimulatedDrop(TestCase):
    """Test the simulated drop data structure"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device_alpha = self.scenario.new_device()
        self.device_beta = self.scenario.new_device()

        self.drop: SimulatedDrop = self.scenario.drop()

    def test_channel_realizations(self) -> None:
        """Channel realizations property should return the correct realizations"""

        self.assertEqual(2, len(self.drop.channel_realizations))

    def test_hdf_serialization_validation(self) -> None:
        """HDF serialization should raise ValueError on invalid scenario arguments"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("group")

        self.drop.to_HDF(group)

        with self.assertRaises(ValueError):
            _ = self.drop.from_HDF(group)

        self.scenario.new_device()

        with self.assertRaises(ValueError):
            _ = SimulatedDrop.from_HDF(group, scenario=self.scenario)

        file.close()

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("group")

        self.drop.to_HDF(group)
        deserialization = SimulatedDrop.from_HDF(group, scenario=self.scenario)

        file.close()

        self.assertIsInstance(deserialization, SimulatedDrop)
        self.assertEqual(self.drop.timestamp, deserialization.timestamp)
        self.assertEqual(self.drop.num_device_transmissions, deserialization.num_device_transmissions)
        self.assertEqual(self.drop.num_device_receptions, deserialization.num_device_receptions)


class TestSimulationScenario(TestCase):
    """Test the Simulation Scenario"""

    def setUp(self) -> None:
        self.seed = 0
        self.device_alpha = SimulatedDevice()
        self.device_beta = SimulatedDevice()

        self.scenario = SimulationScenario(seed=self.seed)
        self.device_alpha = self.scenario.new_device()
        self.device_beta = self.scenario.new_device()

    def test_new_device(self) -> None:
        """Calling new_device should result in a simulated new device being added"""

        new_device = self.scenario.new_device()
        self.assertTrue(self.scenario.device_registered(new_device))

    def test_add_device(self) -> None:
        """Calling add_device should result in the device being added and the channel matrix being expanded"""

        device = Mock()
        self.scenario.add_device(device)

        self.assertTrue(self.scenario.device_registered(device))
        self.assertIs(self.scenario, device.scenario)

    def test_channels_symmetry(self) -> None:
        """Channel matrix should be symmetric"""

        num_added_devices = 3
        for _ in range(num_added_devices):
            self.scenario.add_device(Mock())

        for m in range(self.scenario.num_devices):
            for n in range(self.scenario.num_devices - m):
                self.assertIs(self.scenario.channels[m, n], self.scenario.channels[n, m])

    def test_channel_validation(self) -> None:
        """Querying a channel instance should raise ValueErrors for invalid devices"""

        with self.assertRaises(ValueError):
            _ = self.scenario.channel(self.device_alpha, Mock())

        with self.assertRaises(ValueError):
            _ = self.scenario.channel(Mock(), self.device_beta)

    def test_channel(self) -> None:
        """Querying a channel instance should return the correct channel"""

        channel = self.scenario.channel(self.device_alpha, self.device_beta)
        self.assertIs(self.scenario.channels[0, 1], channel)

    def test_departing_channels_validation(self) -> None:
        """Departing channels should raise a ValueError for invalid devices"""

        with self.assertRaises(ValueError):
            _ = self.scenario.departing_channels(Mock())

    def test_departing_channels(self) -> None:
        """Departing channels should contain the correct channel slice"""

        device = Mock()
        self.scenario.add_device(device)
        self.scenario.channels[0, 2].gain = 0.0

        departing_channels = self.scenario.departing_channels(device, active_only=True)
        expected_departing_channels = self.scenario.channels[1:, 2]
        self.assertCountEqual(expected_departing_channels, departing_channels)

    def test_arriving_channels_validation(self) -> None:
        """Arriving channels should raise a ValueError for invalid devices"""

        with self.assertRaises(ValueError):
            _ = self.scenario.arriving_channels(Mock())

    def test_arriving_channels(self) -> None:
        """Arriving channels should contain the correct channel slice"""

        device = Mock()
        self.scenario.add_device(device)
        self.scenario.channels[2, 0].gain = 0.0

        arriving_channels = self.scenario.arriving_channels(device, active_only=True)
        expected_arriving_channels = self.scenario.channels[2, 1:]
        self.assertCountEqual(expected_arriving_channels, arriving_channels)

    def test_set_channel_validation(self) -> None:
        """Setting a channel should raise a ValueError for invalid device indices"""

        with self.assertRaises(ValueError):
            self.scenario.set_channel(10, 0, Mock())

        with self.assertRaises(ValueError):
            self.scenario.set_channel(0, 10, Mock())

    def test_set_channel(self):
        """Setting a channel should properly integrate the channel into the matrix"""

        device_alpha = self.scenario.new_device()
        device_beta = self.scenario.new_device()

        channel = Mock()
        self.scenario.set_channel(device_alpha, device_beta, channel)

        self.assertIs(channel, self.scenario.channels[2, 3])
        self.assertIs(channel, self.scenario.channels[3, 2])
        self.assertIs(self.scenario, channel.scenario)

    def test_snr_validation(self) -> None:
        """SNR property setter should raise ValueError on arguments less or equal to zero"""

        with self.assertRaises(ValueError):
            self.scenario.snr = -1.0

        with self.assertRaises(ValueError):
            self.scenario.snr = 0.0

        try:
            self.scenario.snr = 0.1234

        except ValueError:
            self.fail()

    def test_snr_setget(self) -> None:
        """SNR property getter should return setter argument"""

        snr = 1.2345
        self.scenario.snr = snr

        self.assertEqual(snr, self.scenario.snr)

        self.scenario.snr = None
        self.assertIsNone(self.scenario.snr)

    def test_snr_type_setget(self) -> None:
        """SNR type property getter should return setter argument"""

        for snr_type in SNRType:
            # Enum set
            self.scenario.snr_type = snr_type
            self.assertEqual(snr_type, self.scenario.snr_type)

            # String set
            self.scenario.snr_type = str(snr_type.name)
            self.assertEqual(snr_type, self.scenario.snr_type)

            # Int Set
            self.scenario.snr_type = snr_type.value
            self.assertEqual(snr_type, self.scenario.snr_type)

    def test_transmit_devices(self) -> None:
        """Transmit devices should return the correct device transmissions"""

        shared_trigger = StaticTrigger()
        self.device_alpha.trigger_model = shared_trigger
        self.device_beta.trigger_model = shared_trigger

        transmissions = self.scenario.transmit_devices()

        self.assertEqual(2, len(transmissions))
        self.assertIs(transmissions[0].trigger_realization, transmissions[1].trigger_realization)

    def test_propagate_validation(self) -> None:
        """Propagate should raise a ValueError for invalid devices"""

        with self.assertRaises(ValueError):
            self.scenario.propagate([Mock() for _ in range(5)])

    def test_process_inputs_validation(self) -> None:
        """Process inputs should raise ValueErrors for invalid arguments"""

        with self.assertRaises(ValueError):
            self.scenario.process_inputs(impinging_signals=[Mock() for _ in range(5)])

        with self.assertRaises(ValueError):
            self.scenario.process_inputs(impinging_signals=[Mock() for _ in range(self.scenario.num_devices)], trigger_realizations=[Mock() for _ in range(5)])

    def test_process_inputs(self) -> None:
        """Process inputs should return the correct device inputs"""

        impinging_signals = [Signal.empty(d.sampling_rate, d.antennas.num_transmit_antennas) for d in self.scenario.devices]
        processed_inputs = self.scenario.process_inputs(impinging_signals=impinging_signals)

        self.assertEqual(2, len(processed_inputs))

    def test_drop(self) -> None:
        """Test the generation of a single drop"""

        drop = self.scenario.drop()

        self.assertEqual(self.scenario.num_devices, drop.num_device_transmissions)
        self.assertEqual(self.scenario.num_devices, drop.num_device_receptions)


class TestSimulationRunner(TestCase):
    """Test the Simulation Runner"""

    def setUp(self) -> None:
        self.seed = 0
        self.scenario = SimulationScenario(seed=self.seed)

        self.sampling_rate = 1e8
        self.num_samples = 100

        self.device_alpha = self.scenario.new_device(sampling_rate=self.sampling_rate)
        self.device_beta = self.scenario.new_device(sampling_rate=self.sampling_rate)

        transmitted_signal = Signal(self.scenario._rng.standard_normal((1, self.num_samples)), self.sampling_rate)
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

        with self.assertRaises(RuntimeError):
            self.runner.process_inputs()

        self.runner.transmit_operators()
        self.runner.generate_outputs()
        self.runner.propagate()
        _ = self.scenario.new_device()

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
        self.modem.waveform_generator = RRCWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)
        self.modem.device = self.device
        self.evaluator = BitErrorEvaluator(self.modem, self.modem)

        self.dimension = self.simulation.new_dimension("snr", [1, 2, 3])
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
        mock_figure = Mock()

        with patch("hermespy.core.monte_carlo.MonteCarloResult.plot", return_value=[mock_figure]), TemporaryDirectory() as temp:
            self.simulation.results_dir = temp
            result = self.simulation.run()

            self.assertIsInstance(result, MonteCarloResult)
            mock_figure.get_figure.assert_called()

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

        self.assertIs(expected_channel, self.simulation.scenario.channels[0, 0])

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.simulation)

    def test_serialization_validation(self) -> None:
        """Test YAML serialization validation"""

        serialization = """
        !<Simulation>
            Devices:
                - !<SimulatedDevice>
                - !<SimulatedDevice>

            Channels:
                - !<Channel>
                - !<Channel>
        """

        factory = Factory()

        with self.assertRaises(RuntimeError):
            _ = factory.from_str(serialization)

    def test_serialization_channel_device_inference(self) -> None:
        """Test YAML serialization with channel device inference"""

        serialization = """
        !<Simulation>
            Devices:
                - !<SimulatedDevice>

            Channels:
                - !<Channel>
        """

        factory = Factory()
        simulation = factory.from_str(serialization)

        self.assertIs(simulation.scenario.devices[0], simulation.scenario.channels[0, 0].alpha_device)
        self.assertIs(simulation.scenario.devices[0], simulation.scenario.channels[0, 0].beta_device)

    def test_serialization_dimension_shorthand(self) -> None:
        """Test YAML serialization with dimension shorthand"""

        serialization = """
        !<Simulation>
            Devices:
                - !<SimulatedDevice>

            Dimensions:
                snr: [1, 2, 3]
        """

        factory = Factory()
        simulation = factory.from_str(serialization)

        self.assertEqual(1, len(simulation.dimensions))
        self.assertEqual("snr", simulation.dimensions[0].dimension)
        self.assertSequenceEqual([1, 2, 3], [p.value for p in simulation.dimensions[0].sample_points])

    def test_pip_pacakges(self) -> None:
        """Test the pip packages property"""

        self.assertCountEqual(["ray", "numpy", "scipy", "matplotlib", "rich", "sparse", "protobuf", "numba"], self.simulation._pip_packages())
