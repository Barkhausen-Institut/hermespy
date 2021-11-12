# -*- coding: utf-8 -*-
"""Test channel equalization over random propagation impulse responses for each waveform."""


import unittest
from copy import deepcopy

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.scenario import Scenario
from hermespy.modem import Transmitter, Receiver, WaveformGeneratorOfdm, WaveformGeneratorPskQam
from hermespy.modem.precoding import SpatialMultiplexing, MMSEqualizer
from hermespy.modem.waveform_generator_ofdm import FrameElement, FrameResource, FrameSymbolSection, ElementType
from hermespy.channel import Channel


class MockChannel(Channel):
    """Mock a propagation channel with a circular additive white gaussian distributed impulse response."""

    def __init__(self, response_std: float = 1.0, num_taps: int = 10, **kwargs) -> None:
        """Initialize mock channel.

        Args:

            response_std (float, optional):
                Standard variation of the impulse response.

            num_taps (int, optional):
                Number of delay taps in the impulse response
        """

        # Init base channel
        Channel.__init__(self, **kwargs)

        self.response_std = response_std
        self.num_taps = num_taps

    def impulse_response(self, timestamps: np.ndarray) -> np.ndarray:

        real = self.random_generator.normal(0, self.response_std,
                                            (len(timestamps), self.receiver.num_antennas,
                                             self.transmitter.num_antennas, self.num_taps))

        imag = self.random_generator.normal(0, self.response_std,
                                            (len(timestamps), self.receiver.num_antennas,
                                             self.transmitter.num_antennas, self.num_taps))

        return (real + imag) * 2 ** -.5


class TestRMSEEqualization(unittest.TestCase):
    """Test channel equalization over a random impulse response for all waveform generators."""

    def setUp(self) -> None:

        self.random_generator = default_rng(53)
        self.scenario = Scenario()

        self.transmitter = Transmitter(num_antennas=2)
        self.transmitter.precoding[0] = SpatialMultiplexing()
        self.scenario.add_transmitter(self.transmitter)

        self.receiver = Receiver(num_antennas=2)
        self.receiver.precoding[0] = SpatialMultiplexing()
        self.receiver.precoding[1] = MMSEqualizer()
        self.scenario.add_receiver(self.receiver)

        self.channel = MockChannel(response_std=1.0, num_taps=5, random_generator=self.random_generator)
        self.scenario.set_channel(0, 0, self.channel)

    def test_quam(self) -> None:
        """Test equalization for the QAM waveform generator."""

        self.transmitter.waveform_generator = WaveformGeneratorPskQam(modulation_order=64, num_data_symbols=30)
        self.receiver.waveform_generator = WaveformGeneratorPskQam(modulation_order=64, num_data_symbols=30)

        transmitted_bits = self.transmitter.generate_data_bits()
        transmitted_signal = self.transmitter.send(data_bits=transmitted_bits)

        propagated_signal, impulse_response = self.channel.propagate(transmitted_signal)

        received_signal = self.receiver.receive([(propagated_signal, 0.0)], 0.0)
        received_bits = self.receiver.demodulate(received_signal, impulse_response, 0.0)

        assert_array_equal(transmitted_bits, received_bits)

    def test_ofdm(self) -> None:
        """Test equalization for the OFDM waveform generator."""

        # OFDM frame configuration
        symbol_elements = [FrameElement(ElementType.DATA) for _ in range(120)]
        resource = FrameResource(repetitions=2, cp_ratio=0.0, elements=symbol_elements)
        section = FrameSymbolSection(pattern=[0])

        self.transmitter.waveform_generator = WaveformGeneratorOfdm(modulation_order=2,
                                                                    resources=[deepcopy(resource)],
                                                                    structure=[deepcopy(section)])
        self.transmitter.topology = np.array([[0.0, 0.0, 0.0]])

        self.receiver.waveform_generator = WaveformGeneratorOfdm(modulation_order=2,
                                                                 resources=[deepcopy(resource)],
                                                                 structure=[deepcopy(section)])
        self.receiver.topology = np.array([[0.0, 0.0, 0.0]])

        transmitted_bits = self.transmitter.generate_data_bits()
        transmitted_signal = self.transmitter.send(data_bits=transmitted_bits)

        propagated_signal, impulse_response = self.channel.propagate(transmitted_signal)

        impulse_response *= np.sqrt(transmitted_signal.shape[1])
        received_signal = self.receiver.receive([(propagated_signal, 0.0)], 0.0)

        symbols, impulse, noise = self.receiver.waveform_generator.demodulate(received_signal[0, ::],
                                                                              impulse_response[:, 0, :, 0], 0.0)

        received_bits = self.receiver.demodulate(received_signal, impulse_response, 0.0)

        assert_array_equal(transmitted_bits, received_bits)

