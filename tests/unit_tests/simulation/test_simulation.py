# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable."""

from unittest import TestCase
from unittest.mock import Mock, patch

import ray

from hermespy.simulation.simulation import Simulation, SimulationActor, SimulationRunner, SimulationScenario, SNRType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulationScenario(TestCase):
    """Test the Simulation Scenario."""

    def setUp(self) -> None:

        self.seed = 0
        self.device_alpha = Mock()
        self.device_beta = Mock()

        self.scenario = SimulationScenario(seed=self.seed)
        self.scenario.add_device(self.device_alpha)
        self.scenario.add_device(self.device_beta)

    def test_new_device(self) -> None:
        """Calling new_device should result in a simulated new device being added."""

        new_device = self.scenario.new_device()
        self.assertTrue(self.scenario.device_registered(new_device))

    def test_add_device(self) -> None:
        """Calling add_device should result in the device being added and the channel matrix being expanded."""

        device = Mock()
        self.scenario.add_device(device)

        self.assertTrue(self.scenario.device_registered(device))
        self.assertIs(self.scenario, device.scenario)

    def test_channels_symmetry(self) -> None:
        """Channel matrix should be symmetric."""

        num_added_devices = 3
        for _ in range(num_added_devices):
            self.scenario.add_device(Mock())

        for m in range(self.scenario.num_devices):
            for n in range(self.scenario.num_devices - m):
                self.assertIs(self.scenario.channels[m, n], self.scenario.channels[n, m])

    def test_departing_channels(self) -> None:
        """Departing channels should contain the correct channel slice."""

        device = Mock()
        self.scenario.add_device(device)
        self.scenario.channels[0, 2].active = False

        departing_channels = self.scenario.departing_channels(device, active_only=True)
        expected_departing_channels = (self.scenario.channels[1:, 2])
        self.assertCountEqual(expected_departing_channels, departing_channels)

    def test_arriving_channels(self) -> None:
        """Arriving channels should contain the correct channel slice."""

        device = Mock()
        self.scenario.add_device(device)
        self.scenario.channels[2, 0].active = False

        arriving_channels = self.scenario.departing_channels(device, active_only=True)
        expected_departing_channels = (self.scenario.channels[2, 1:])
        self.assertCountEqual(expected_departing_channels, arriving_channels)

    def test_set_channel(self):
        """Setting a channel should properly integrate the channel into the matrix."""

        device_alpha = Mock()
        device_beta = Mock()
        self.scenario.add_device(device_alpha)
        self.scenario.add_device(device_beta)

        channel = Mock()
        self.scenario.set_channel(2, 3, channel)

        self.assertIs(channel, self.scenario.channels[2, 3])
        self.assertIs(channel, self.scenario.channels[3, 2])
        self.assertIs(self.scenario, channel.random_mother)
        self.assertIs(self.scenario, channel.scenario)

    def test_snr_setget(self) -> None:
        """SNR property getter should return setter argument."""

        snr = 1.2345
        self.scenario.snr = snr

        self.assertEqual(snr, self.scenario.snr)

    def test_snr_validation(self) -> None:
        """SNR property setter should raise ValueError on arguments less or equal to zero."""

        with self.assertRaises(ValueError):
            self.scenario.snr = -1.

        with self.assertRaises(ValueError):
            self.scenario.snr = 0.

        try:
            self.scenario.snr = 0.1234

        except ValueError:
            self.fail()
            
    def test_snr_type_setget(self) -> None:
        """SNR type property getter should return setter argument."""

        for snr_type in SNRType:

            # Enum set
            self.scenario.snr_type = snr_type
            self.assertEqual(snr_type, self.scenario.snr_type)

            # String set
            self.scenario.snr_type = str(snr_type.name)
            self.assertEqual(snr_type, self.scenario.snr_type)


class TestSimulationRunner(TestCase):
    """Test the Simulation Runner."""

    def setUp(self) -> None:

        self.seed = 0
        self.device_alpha = Mock()
        self.device_alpha.max_frame_duration = 1.
        self.device_beta = Mock()
        self.device_beta.max_frame_duration = 1.

        self.transmitter_alpha = Mock()
        self.transmitter_beta = Mock()
        self.receiver_alpha = Mock()
        self.receiver_beta = Mock()

        self.device_alpha.transmitters = [self.transmitter_alpha]
        self.device_beta.transmitters = [self.transmitter_beta]
        self.device_alpha.receivers = [self.receiver_alpha]
        self.device_beta.receivers = [self.receiver_beta]

        self.scenario = SimulationScenario(seed=self.seed)
        self.scenario.add_device(self.device_alpha)
        self.scenario.add_device(self.device_beta)
        
        for m in range(self.scenario.num_devices):
            for n in range(self.scenario.num_devices):

                channel = Mock()
                channel.propagate.return_value = Mock(), Mock(), Mock()
                self.scenario.set_channel(m, n, channel)
                

        self.runner = SimulationRunner(self.scenario)

    def test_transmit_operators(self) -> None:

        self.runner.transmit_operators()

        self.assertTrue(self.transmitter_alpha.transmit.called)
        self.assertTrue(self.transmitter_beta.transmit.called)

    def test_transmit_devices(self) -> None:
        """Transmit devices should call each device's transmit function."""

        self.runner.transmit_devices()

        for device in self.scenario.devices:
            self.assertTrue(device.transmit.called)

    def test_propagate(self) -> None:
        """A single propagation should result in each channel being sampled once."""

        self.runner.transmit_operators()
        self.runner.transmit_devices()
        self.runner.propagate()

        for channel in self.scenario.channels.flat:
            self.assertEqual(1, channel.propagate.call_count)

    def test_receive_devices(self) -> None:
        """Test receive devices stage callback execution"""
        
        self.runner.transmit_operators()
        self.runner.transmit_devices()
        self.runner.propagate()
        self.runner.receive_devices()
        
        self.device_alpha.receive.assert_called()
        self.device_beta.receive.assert_called()
        
    def test_receiver_operators(self) -> None:
        """Test receive operators stage callback execution"""
        
        self.runner.receive_operators()
        
        self.receiver_alpha.receive.assert_called()
        self.receiver_beta.receive.assert_called()

class TestSimulationActor(TestCase):
    """Test the Simulation Actor."""

    def setUp(self) -> None:

        self.seed = 0
        self.device_alpha = Mock()
        self.device_beta = Mock()

        self.scenario = SimulationScenario(seed=self.seed)
        self.scenario.add_device(self.device_alpha)
        self.scenario.add_device(self.device_beta)

        self.dimensions = []
        self.evaluator = Mock()

        self.actor = SimulationActor.remote((self.scenario, self.dimensions, [self.evaluator]))

    def test_sample(self) -> None:
        """"""

        tx_alpha = Mock()
        tx_beta = Mock()
        propagation_matrix = Mock()
        rx_device_signal = Mock()


class TestSimulation(TestCase):
    """Test the simulation executable, the base class for simulation operations."""

    def setUp(self) -> None:

        self.simulation = Simulation()
        
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=1)
            
    @classmethod
    def tearDownClass(cls):
        
        ray.shutdown()

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass
