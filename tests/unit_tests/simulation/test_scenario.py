# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase
from unittest.mock import Mock

from hermespy.core import Signal
from hermespy.simulation.noise import NoiseLevel, NoiseModel
from hermespy.simulation.scenario import SimulationScenario
from hermespy.simulation.simulated_device import SimulatedDevice, StaticTrigger
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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

    def test_noise_level_setget(self) -> None:
        """Noise level property getter should return setter argument"""

        noise_level = Mock(spec=NoiseLevel)
        self.scenario.noise_level = noise_level
        self.assertIs(noise_level, self.scenario.noise_level)

        self.scenario.noise_level = None
        self.assertIsNone(self.scenario.noise_level)

    def test_noise_model_setget(self) -> None:
        """Noise model property getter should return setter argument"""

        noise_model = Mock(spec=NoiseModel)
        self.scenario.noise_model = noise_model
        self.assertIs(noise_model, self.scenario.noise_model)
        self.assertIs(self.scenario, noise_model.random_mother)

        self.scenario.noise_model = None
        self.assertIsNone(self.scenario.noise_model)
        
    def test_generate_outputs_validation(self) -> None:
        """Generate outputs should raise ValueErrors for invalid arguments"""
        
        # Invalid number of operator transmissions
        with self.assertRaises(ValueError):
            self.scenario.generate_outputs([Mock() for _ in range(1 + self.scenario.num_devices)],)
            
        # Empty list of trigger realizations
        with self.assertRaises(ValueError):
            self.scenario.generate_outputs([Mock() for _ in range(self.scenario.num_devices)], [])
        
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

        impinging_signals = [Signal.Empty(d.sampling_rate, d.antennas.num_transmit_antennas) for d in self.scenario.devices]
        processed_inputs = self.scenario.process_inputs(impinging_signals=impinging_signals)

        self.assertEqual(2, len(processed_inputs))

    def test_drop(self) -> None:
        """Test the generation of a single drop"""

        drop = self.scenario.drop()

        self.assertEqual(self.scenario.num_devices, drop.num_device_transmissions)
        self.assertEqual(self.scenario.num_devices, drop.num_device_receptions)

    def test_visualize(self) -> None:
        """Visualize the scenario should not raise any exceptions"""

        with SimulationTestContext():
            self.scenario.visualize()
