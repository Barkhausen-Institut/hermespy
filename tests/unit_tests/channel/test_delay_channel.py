# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal
from scipy.constants import speed_of_light

from hermespy.channel import InterpolationMode, SpatialDelayChannel, SpatialDelayChannelRealization, RandomDelayChannel, RandomDelayChannelRealization
from hermespy.channel.delay import DelayChannelBase, DelayChannelRealization
from hermespy.core import Signal, Transformation
from hermespy.simulation import SimulatedDevice
from hermespy.tools import amplitude_path_loss
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestDelayChannelRealization(TestCase):
    def _init_realization(self, *args, **kwargs) -> DelayChannelRealization:
        ...

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.carrier_frequency = 1.234e9

        self.alpha_device = SimulatedDevice(pose=Transformation.From_Translation(np.array([0, 0, 0])), carrier_frequency=self.carrier_frequency)
        self.beta_device = SimulatedDevice(pose=Transformation.From_Translation(np.array([0, 0, 10])), carrier_frequency=self.carrier_frequency)
        self.delay = 1.234
        self.gain = 0.987

        self.realization = self._init_realization(alpha_device=self.alpha_device, beta_device=self.beta_device, delay=self.delay, gain=self.gain, model_propagation_loss=True, interpolation_mode=InterpolationMode.NEAREST)

    def test_properties(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.alpha_device, self.realization.alpha_device)
        self.assertIs(self.beta_device, self.realization.beta_device)
        self.assertEqual(self.delay, self.realization.delay)
        self.assertEqual(self.gain, self.realization.gain)

    def test_propagate_validation(self) -> None:
        """Propagation routine should raise RuntimeError for base band transmissions"""

        test_signal = Signal(np.ones((1, 1), dtype=np.complex_), sampling_rate=1.0, carrier_frequency=0.0)

        with self.assertRaises(RuntimeError):
            _ = self.realization.propagate(test_signal)

    def test_propagate(self) -> None:
        """Signal propagation should result in the correct path losses and delays"""

        num_samples = 10
        expected_delay_in_samples = 3
        sampling_rate = expected_delay_in_samples / self.delay
        expected_signal_scale = self.gain * amplitude_path_loss(self.carrier_frequency, self.delay * speed_of_light)
        test_signal = Signal(np.ones((1, num_samples), dtype=np.complex_), sampling_rate=sampling_rate, carrier_frequency=self.carrier_frequency)

        propagation = self.realization.propagate(test_signal)

        self.assertAlmostEqual(num_samples * expected_signal_scale, np.sum(propagation.signal.samples))
        self.assertEqual(expected_delay_in_samples + num_samples, propagation.signal.num_samples)

    def test_propagate_state(self) -> None:
        """Propagation should result in a signal with the correct number of samples"""

        sampling_rate = 1e6
        samples = self.rng.standard_normal((self.alpha_device.antennas.num_transmit_antennas, 100)) + 1j * self.rng.standard_normal((self.alpha_device.antennas.num_transmit_antennas, 100))
        signal = Signal(samples, sampling_rate, self.carrier_frequency)

        signal_propagation = self.realization.propagate(signal)
        state_propagation = self.realization.state(self.alpha_device, self.beta_device, 0.0, sampling_rate, signal.num_samples, 1 + signal_propagation.signal.num_samples - signal.num_samples).propagate(signal)

        assert_array_almost_equal(signal_propagation.signal.samples, state_propagation.samples)


class TestSpatialDelayChannelRealization(_TestDelayChannelRealization):
    """Test the spatial delay channel realization"""

    def _init_realization(self, *args, **kwargs) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization(*args, **kwargs)


class TestRandomDelayChannelRealization(_TestDelayChannelRealization):
    """Test the random delay channel realization"""

    def _init_realization(self, *args, **kwargs) -> RandomDelayChannelRealization:
        return RandomDelayChannelRealization(*args, **kwargs)


class _TestDelayChannelBase(TestCase):
    """Test base class of all delay channels"""

    def _init_channel(self, *args, **kwargs) -> DelayChannelBase:
        ...

    def setUp(self) -> None:
        self.carrier_frequency = 1.234e9
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([0, 0, 0])))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([0, 0, 10])))

        self.channel = self._init_channel(alpha_device=self.alpha_device, beta_device=self.beta_device)

    def test_properties(self) -> None:
        """Properties should be properly initialized"""

        self.assertIs(self.alpha_device, self.channel.alpha_device)
        self.assertIs(self.beta_device, self.channel.beta_device)

    def test_realize(self) -> None:
        """Test channel realization"""

        realization = self.channel.realize()

        self.assertIs(self.alpha_device, realization.alpha_device)
        self.assertIs(self.beta_device, realization.beta_device)
        self.assertEqual(self.channel.gain, realization.gain)

    def test_recall_realization(self) -> None:
        """Test realization recall"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("g")

        expected_realization = self.channel.realize()
        expected_realization.to_HDF(group)

        recalled_realization = self.channel.recall_realization(group)
        file.close()

        self.assertIsInstance(recalled_realization, type(expected_realization))
        self.assertEqual(expected_realization.delay, recalled_realization.delay)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.channel.Channel.alpha_device", new_callable=PropertyMock) as transmitter_mock, patch("hermespy.channel.Channel.beta_device", new_callable=PropertyMock) as receiver_mock, patch("hermespy.channel.Channel.random_mother", new_callable=PropertyMock) as random_mock:
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None

            test_yaml_roundtrip_serialization(self, self.channel)

    def test_recall_realization(self) -> None:
        """Test realization recall from HDF"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("g")

        expected_realization = self.channel.realize()
        expected_realization.to_HDF(group)

        recalled_realization = self.channel.recall_realization(group)
        file.close()

        self.assertEqual(expected_realization.delay, recalled_realization.delay)


class TestSpatialDelayChannel(_TestDelayChannelBase):
    """Test the spatial delay channel"""

    def _init_channel(self, *args, **kwargs) -> SpatialDelayChannel:
        return SpatialDelayChannel(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()

        self.expected_delay = 1.4567
        self.alpha_device.position = np.zeros(3)
        self.beta_device.position = np.ones(3) / np.sqrt(3) * self.expected_delay * speed_of_light

    def test_delay_realization_validation(self) -> None:
        """Delay realization should raise RuntimeErrors on invalid internal states"""

        self.channel.alpha_device = None
        with self.assertRaises(RuntimeError):
            self.channel._realize_delay()

    def test_delay_realization(self) -> None:
        """Delay realization should yield the correct time delay"""

        realization = self.channel.realize()
        self.assertAlmostEqual(self.expected_delay, realization.delay, places=7)

    def test_power_loss(self) -> None:
        """Propagated signals should loose power according to the free space propagation loss"""

        # Assert free space propagation power loss
        power_signal = Signal(np.zeros((self.alpha_device.num_antennas, 10)), self.alpha_device.sampling_rate, self.alpha_device.carrier_frequency)
        power_signal.samples[0, :] = np.ones(10)

        initial_energy = np.sum(power_signal.energy)
        expected_received_energy = initial_energy * amplitude_path_loss(self.alpha_device.carrier_frequency, self.expected_delay * speed_of_light) ** 2

        propagation = self.channel.propagate(power_signal)
        received_energy = np.mean(propagation.signal.energy)

        self.assertAlmostEqual(expected_received_energy, received_energy)

        # Assert no power loss (flag disabled)
        self.channel.model_propagation_loss = False
        propagation = self.channel.propagate(power_signal)

        self.assertAlmostEqual(initial_energy, np.mean(propagation.signal.energy))


class TestRandomDelayChannel(_TestDelayChannelBase):
    def _init_channel(self, *args, **kwargs) -> SpatialDelayChannel:
        return RandomDelayChannel(0.0, *args, seed=42, **kwargs)

    def test_delay_validation(self) -> None:
        """Invalid delay arguments should raise ValueErrors"""

        with self.assertRaises(ValueError):
            self.channel.delay = -1.0

        with self.assertRaises(ValueError):
            self.channel.delay = (-1.0, 1.0)

        with self.assertRaises(ValueError):
            self.channel.delay = (2.0, 1.0)

        with self.assertRaises(ValueError):
            self.channel.delay = (1.0, 2.0, 3.0)

        with self.assertRaises(ValueError):
            self.channel.delay = "wrong param"

    def test_constant_delay_realization(self) -> None:
        """Setting the delay as a scalar should realize a constant delay"""

        expected_delay = 1.0
        self.channel.delay = expected_delay

        for _ in range(3):
            realization = self.channel.realize()
            self.assertEqual(expected_delay, realization.delay)

    def test_random_delay_realization(self) -> None:
        """Setting the delay as a scalar should realize a draw from the uniform distribution"""

        min_delay = 10
        max_delay = 11

        self.channel.delay = min_delay, max_delay

        for _ in range(5):
            realization = self.channel.realize()
            self.assertTrue(min_delay <= realization.delay <= max_delay)

    def test_delay_realization_validation(self) -> None:
        """Delay realization should raise a runtime error on invalid delay configurations"""

        with patch("hermespy.channel.delay.RandomDelayChannel.delay", new_callable=PropertyMock) as delay_patch:
            delay_patch.return_value = "wrong value"

            with self.assertRaises(RuntimeError):
                _ = self.channel.realize()


del _TestDelayChannelRealization
del _TestDelayChannelBase
