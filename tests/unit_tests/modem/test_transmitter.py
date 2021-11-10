# -*- coding: utf-8 -*-
"""Test HermesPy transmit modem class."""

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.random as rnd
from math import ceil

from hermespy.modem import Transmitter

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestTransmitter(unittest.TestCase):
    """Test the transmit modem implementation."""

    def setUp(self) -> None:

        self.power = 0.9
        self.scenario = Mock()
        self.scenario.random_generator = rnd.default_rng(0)
        self.scenario.sampling_rate = 1e3

        self.waveform_generator = Mock()
        self.waveform_generator.symbols_per_frame = 100
        self.waveform_generator.samples_in_frame = 100
        self.waveform_generator.bits_per_frame = 100
        self.waveform_generator.frame_duration = 100 / self.scenario.sampling_rate
        self.waveform_generator.map = lambda bits: bits
        self.waveform_generator.modulate = lambda symbols, timestamps: symbols

        self.encoder_manager = Mock()
        self.encoder_manager.encode = lambda bits, num_bits: np.append(bits, np.zeros(num_bits - len(bits)))
        self.encoder_manager.required_num_data_bits = lambda num_bits: num_bits

        self.precoding = Mock()
        self.precoding.encode = lambda symbols: np.array([symbols], dtype=complex)
        self.precoding.rate = 1.0

        self.bits_source = Mock()
        self.bits_source.get_bits = lambda num_bits: np.repeat(np.array([0, 1]), int(ceil(.5 * num_bits)))[:num_bits]

        self.transmitter = Transmitter(power=self.power, scenario=self.scenario, waveform=self.waveform_generator,
                                       encoding=self.encoder_manager, precoding=self.precoding,
                                       bits_source=self.bits_source)

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored."""

        self.assertEqual(self.power, self.transmitter.power)
        self.assertEqual(self.scenario, self.transmitter.scenario)
        self.assertEqual(self.waveform_generator, self.transmitter.waveform_generator)
        self.assertEqual(self.encoder_manager, self.transmitter.encoder_manager)
        self.assertEqual(self.precoding, self.transmitter.precoding)
        self.assertEqual(self.bits_source, self.transmitter.bits_source)

    def test_power_setget(self) -> None:
        """Transmit power property getter should return setter argument."""

        power = 0.5
        self.transmitter.power = power

        self.assertEqual(power, self.transmitter.power)

    def test_power_validation(self) -> None:
        """Transmit power property setter should raise ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.transmitter.power = -0.2

    def test_send_default(self) -> None:
        """Test the send routine with default parameters."""

        baseband_signal = self.transmitter.send()

    def test_index(self) -> None:
        """Index property should return the transmitter's position within the scenario's transmitter list."""

        self.scenario.transmitters = [Mock(), Mock(), self.transmitter]
        self.assertIs(2, self.transmitter.index)

    def test_paired_modems(self) -> None:
        """The paired modems property should return receiver objects from channels."""

        channel_list = [Mock() for _ in range(5)]
        expected_paired_modems = [Mock() for _ in range(5)]
        for channel, modem in zip(channel_list, expected_paired_modems):
            channel.receiver = modem

        self.scenario.departing_channels = lambda transmitter, only_active: channel_list

        paired_modems = self.transmitter.paired_modems
        self.assertCountEqual(expected_paired_modems, paired_modems)

    def test_generate_data_bits_length(self) -> None:
        """The number of data bits generated must be the number of required bits for one frame."""

        self.assertEqual(self.transmitter.num_data_bits_per_frame, len(self.transmitter.generate_data_bits()))