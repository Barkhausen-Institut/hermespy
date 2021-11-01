# -*- coding: utf-8 -*-
"""Test HermesPy scenario description class."""

import unittest
import numpy as np
import numpy.random as rnd
from typing import List
from unittest.mock import Mock
from itertools import product
from numpy.testing import assert_array_equal

from scenario.scenario import Scenario

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class TestScenario(unittest.TestCase):
    """Test the base class for describing a full wireless communication scenario."""

    def setUp(self) -> None:

        self.random_generator = rnd.default_rng(0)
        self.drop_duration = 1e-3
        self.sampling_rate = 4e6
        self.scenario = Scenario(drop_duration=self.drop_duration,
                                 sampling_rate=self.sampling_rate,
                                 random_generator=self.random_generator)

        self.num_transmitters = 2
        self.num_receivers = 2

        self.transmitters: List[Mock] = []
        for t in range(self.num_transmitters):

            modem = Mock()

            # Mock waveform generator max frame duration property
            modem.waveform_generator.max_frame_duration = (1+t) * 1e-4
            modem.generate_data_bits.return_value = np.ones(1+t)
            modem.encoder_manager.block_size = 1 + t

            self.transmitters.append(modem)
            self.scenario.add_transmitter(modem)

        self.receivers: List[Mock] = []
        for r in range(self.num_receivers):

            modem = Mock()
            modem.encoder_manager.block_size = 1 + r

            self.receivers.append(modem)
            self.scenario.add_receiver(modem)

    def test_init(self) -> None:
        """Object initialization should properly set all argument parameters."""

        self.assertEqual(self.drop_duration, self.scenario.drop_duration)
        self.assertIs(self.random_generator, self.scenario.random_generator)
        self.assertIs(self.sampling_rate, self.scenario.sampling_rate)

    def test_receivers_get(self) -> None:
        """Receivers property should return a list of all added receiver modems."""

        self.assertCountEqual(self.receivers, self.scenario.receivers)

    def test_transmitters_get(self) -> None:
        """Transmitters property should return a list of all added transmitter modems."""

        self.assertCountEqual(self.transmitters, self.scenario.transmitters)

    def test_num_receivers(self) -> None:
        """The number of receivers property should return the correct number of registered receive modems."""

        for t in range(1, 3):

            self.scenario.add_receiver(Mock())
            self.assertEqual(self.num_receivers + t, self.scenario.num_receivers)

    def test_num_transmitters(self) -> None:
        """The number of transmitters property should return the correct number of registered transmit modems."""

        for r in range(1, 3):
            self.scenario.add_transmitter(Mock())
            self.assertEqual(self.num_transmitters + r, self.scenario.num_transmitters)

    def test_channel_search(self) -> None:
        """The channel function should return the specific channel instance,
        connecting the transmitter argument to the receiver argument."""

        for transmitter, receiver in product(self.transmitters, self.receivers):

            channel = self.scenario.channel(transmitter, receiver)
            self.assertEqual(transmitter, channel.transmitter)
            self.assertEqual(receiver, channel.receiver)

    def test_channel_validation(self) -> None:
        """The channel function should raise ValueErrors on invalid arguments."""

        if self.num_receivers > 0:
            with self.assertRaises(ValueError):
                _ = self.scenario.channel(self.transmitters[0], Mock())

        if self.num_transmitters > 0:
            with self.assertRaises(ValueError):
                _ = self.scenario.channel(Mock(), self.receivers[0])

    def test_departing_channels(self) -> None:
        """Departing channels search function should return all channels connected to the transmitter argument."""

        for t, transmitter in enumerate(self.transmitters):

            channels: List[Mock] = []
            for r, receiver in enumerate(self.receivers):

                channel = Mock()
                self.scenario.set_channel(t, r, channel)
                channels.append(channel)

            departing_channels = self.scenario.departing_channels(transmitter)
            self.assertCountEqual(channels, departing_channels)

    def test_departing_channels_validation(self) -> None:
        """Departing channels search function should raise a ValueError
        if the transmitter argument is not registered. with the scenario."""

        with self.assertRaises(ValueError):
            _ = self.scenario.departing_channels(Mock())

    def test_arriving_channels(self) -> None:
        """Arriving channels search function should return all channels connected to the receiver argument."""

        for r, receiver in enumerate(self.receivers):

            channels: List[Mock] = []
            for t, transmitter in enumerate(self.transmitters):

                channel = Mock()
                self.scenario.set_channel(t, r, channel)
                channels.append(channel)

            arriving_channels = self.scenario.arriving_channels(receiver)
            self.assertCountEqual(channels, arriving_channels)

    def test_arriving_channels_validation(self) -> None:
        """Arriving channels search function should raise a ValueError
        if the receiver argument is not registered. with the scenario."""

        with self.assertRaises(ValueError):
            _ = self.scenario.arriving_channels(Mock())

    def test_set_channel(self) -> None:
        """A channel should be set at the proper matrix coordinates
        and return the respective transmitter and receiver modem handles"""

        channel = Mock()
        transmitter_index = self.num_transmitters - 1
        receiver_index = self.num_receivers - 1
        self.scenario.set_channel(transmitter_index, receiver_index, channel)

        self.assertIs(channel, self.scenario.channels[transmitter_index, receiver_index])
        self.assertIs(channel.transmitter, self.transmitters[transmitter_index])
        self.assertIs(channel.receiver, self.receivers[receiver_index])

    def test_set_channel_validation(self) -> None:
        """Setting a channel should raise ArgumentErrors on invalid index arguments."""

        with self.assertRaises(ValueError):
            self.scenario.set_channel(-1, self.num_receivers-1, Mock())

        with self.assertRaises(ValueError):
            self.scenario.set_channel(self.num_transmitters, self.num_receivers - 1, Mock())

        with self.assertRaises(ValueError):
            self.scenario.set_channel(self.num_transmitters - 1, -1, Mock())

        with self.assertRaises(ValueError):
            self.scenario.set_channel(self.num_transmitters - 1, self.num_receivers, Mock())

    def test_add_receiver(self) -> None:
        """Adding a receiver should set the receiver's scenario attribute,
        as well as adding an entry in the second channel matrix dimension."""

        receiver = Mock()

        self.scenario.add_receiver(receiver)
        self.assertIs(self.scenario, receiver.scenario)

        new_channels = self.scenario.arriving_channels(receiver)
        for channel in new_channels:
            self.assertIs(receiver, channel.receiver)

    def test_add_transmitter(self) -> None:
        """Adding a transmitter should set the transmitter's scenario attribute,
        as well as adding a column in the first channel matrix dimension."""

        transmitter = Mock()

        self.scenario.add_transmitter(transmitter)
        self.assertIs(self.scenario, transmitter.scenario)

        new_channels = self.scenario.departing_channels(transmitter)
        for channel in new_channels:
            self.assertIs(transmitter, channel.transmitter)

    def test_remove_modem(self) -> None:
        """Removing a modem should result in the deletion of all channels within the respective row or column
        of the channel matrix."""

        if self.num_receivers > 0:

            self.scenario.remove_modem(self.receivers[0])

            self.assertEqual(self.num_receivers - 1, self.scenario.channels.shape[1])
            self.assertNotIn(self.receivers[0], self.scenario.receivers)

            for channel in self.scenario.channels.flatten():
                self.assertIsNot(self.receivers[0], channel.receiver)
                
        if self.num_transmitters > 0:

            self.scenario.remove_modem(self.transmitters[0])

            self.assertEqual(self.num_transmitters - 1, self.scenario.channels.shape[1])
            self.assertNotIn(self.transmitters[0], self.scenario.transmitters)

            for channel in self.scenario.channels.flatten():
                self.assertIsNot(self.transmitters[0], channel.transmitter)

    def test_remove_modem_assert(self) -> None:
        """Removing a modem not registered with the scenario should result in a ValueError."""

        with self.assertRaises(ValueError):
            self.scenario.remove_modem(Mock())

    def test_drop_duration_setget(self) -> None:
        """The drop duration property getter should return the setter argument,"""

        drop_duration = 12345
        self.scenario.drop_duration = drop_duration

        self.assertEqual(drop_duration, self.scenario.drop_duration)

    def test_drop_duration_validation(self) -> None:
        """The drop duration property setter should raise a ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.scenario.drop_duration = -1

        try:
            self.scenario.drop_duration = 0.0

        except ValueError:
            self.fail("Setting a drop duration of zero should not result in an error throw")

    def test_drop_duration_computation(self) -> None:
        """If the drop duration is set to zero,
        the property getter should return the maximum frame duration as drop duration."""

        max_frame_duration = self.num_transmitters * 1e-4     # Results from the setUp transmit mock
        self.scenario.drop_duration = 0.0

        self.assertEquals(max_frame_duration, self.scenario.drop_duration)

    def test_generate_data_bits(self) -> None:
        """The data bit generation routine should create sets of source bits required by all registered
        transmitters in order to compute a single data frame."""

        expected_data_bits: List[np.ndarray] = []
        for t in range(self.num_transmitters):
            expected_data_bits.append(np.ones(1+t))     # From Mock generation

        data_bits = self.scenario.generate_data_bits()
        for b, expected_bits in enumerate(expected_data_bits):
            assert_array_equal(expected_bits, data_bits[b])

    def test_transmit_block_sizes(self) -> None:
        """Transmit blocks sizes property should return a list of all transmitters respective block sizes."""

        expected_block_sizes: List[int] = []
        for transmitter in self.transmitters:
            expected_block_sizes.append(transmitter.encoder_manager.block_size)

        self.assertCountEqual(expected_block_sizes, self.scenario.transmit_block_sizes)

    def test_receive_block_sizes(self) -> None:
        """Receive blocks sizes property should return a list of all receivers respective block sizes."""

        expected_block_sizes: List[int] = []
        for receiver in self.receivers:
            expected_block_sizes.append(receiver.encoder_manager.block_size)

        self.assertCountEqual(expected_block_sizes, self.scenario.receive_block_sizes)

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument."""

        sampling_rate = 1.0
        self.scenario.sampling_rate = sampling_rate

        self.assertEqual(sampling_rate, self.scenario.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on arguments zero or negative."""

        with self.assertRaises(ValueError):
            self.scenario.sampling_rate = 0.0

        with self.assertRaises(ValueError):
            self.scenario.sampling_rate = -1.0

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass
