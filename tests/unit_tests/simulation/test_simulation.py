# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable."""

from unittest import TestCase
from typing import Optional
from unittest.mock import Mock, patch

import ray

from hermespy.core import ChannelStateInformation, Transmitter, Transmission, Receiver, Reception, Signal
from hermespy.simulation.simulation import Simulation, SimulationActor, SimulationRunner, SimulationScenario, SNRType
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulationScenario(TestCase):
    """Test the Simulation Scenario."""

    def setUp(self) -> None:

        self.seed = 0

        self.scenario = SimulationScenario(seed=self.seed)
        self.device_alpha = self.scenario.new_device()
        self.device_beta = self.scenario.new_device()

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
        
    def test_drop(self) -> None:
        """Test the generation of a single drop"""
        
        drop = self.scenario.drop()


class MockTransmitter(Transmitter):
    """Mock transmitter for testing purposes."""

    def transmit(self, duration: float = 0) -> Transmission:
        
        signal = Signal.empty(self.sampling_rate, self.device.num_antennas, 1, carrier_frequency=self.carrier_frequency)
        return Transmission(signal)

    @property
    def frame_duration(self) -> float:
        return 1.

    @property
    def sampling_rate(self) -> float:
        return 1.


class MockReceiver(Receiver):
    """Mock receiver for testing purposes."""

    def _receive(self, signal: Signal, csi: Optional[ChannelStateInformation] = None, cache: bool = True) -> Reception:
        return Reception(signal.resample(self.sampling_rate))

    @property
    def frame_duration(self) -> float:
        return 1.

    @property
    def sampling_rate(self) -> float:
        return 1.

    def _noise_power(self, strength, snr_type=...) -> float:
        return strength


class TestSimulationRunner(TestCase):
    """Test the Simulation Runner."""

    def setUp(self) -> None:

        self.seed = 0
        self.scenario = SimulationScenario(seed=self.seed)
        self.device_alpha = self.scenario.new_device()
        self.device_beta = self.scenario.new_device()

        self.transmitter_alpha = MockTransmitter()
        self.transmitter_beta = MockTransmitter()
        self.receiver_alpha = MockReceiver()
        self.receiver_beta = MockReceiver()

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

        self.actor = SimulationActor.remote((self.scenario, self.dimensions, [self.evaluator]), 0)

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
       
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.simulation) 
