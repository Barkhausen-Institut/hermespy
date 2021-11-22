# -*- coding: utf-8 -*-
"""Test HermesPy Orthogonal Frequency Division Multiplexing Waveform Generation."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.random import default_rng
from scipy.constants import pi

from hermespy.modem import WaveformGeneratorOfdm, FrameSymbolSection, FrameResource
from hermespy.modem.waveform_generator_ofdm import FrameElement, ElementType


class TestWaveformGeneratorOFDM(unittest.TestCase):
    """Test Orthogonal Frequency Division Multiplexing Waveform Generator."""

    def setUp(self) -> None:

        self.subcarrier_spacing = 1e3
        self.num_subcarriers = 100

        self.random_generator = default_rng(42)

        self.modem = Mock()
        self.modem.random_generator = self.random_generator
        self.modem.scenario.sampling_rate = 4 * self.num_subcarriers * self.subcarrier_spacing

        self.waveform_generator = WaveformGeneratorOfdm(subcarrier_spacing=self.subcarrier_spacing, modem=self.modem)

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.waveform_generator.modem)
        self.assertEqual(self.subcarrier_spacing, self.waveform_generator.subcarrier_spacing)

    def test_add_resource(self) -> None:
        """Added resources should be properly appended to the resource list."""

        resource = Mock()
        self.waveform_generator.add_resource(resource)

        self.assertIn(resource, self.waveform_generator.resources)

    def test_add_section(self) -> None:
        """Added sections should be properly appended to the section list."""

        section = Mock()
        self.waveform_generator.add_section(section)

        self.assertIn(section, self.waveform_generator.structure)
        self.assertIs(self.waveform_generator, section.frame)

    def test_subcarrier_spacing_setget(self) -> None:
        """Subcarrier spacing property getter should return setter argument."""

        spacing = 123
        self.waveform_generator.subcarrier_spacing = spacing

        self.assertEqual(spacing, self.waveform_generator.subcarrier_spacing)

    def test_subcarrier_spacing_assert(self) -> None:
        """Subcarrier spacing property setter should raise ValueError on arguments zero or smaller."""

        with self.assertRaises(ValueError):
            self.waveform_generator.subcarrier_spacing = -1.

        with self.assertRaises(ValueError):
            self.waveform_generator.subcarrier_spacing = 0.

    def test_section_demodulation_symbol_transformation(self) -> None:
        """Data and noise signals should be properly transformed to frequency domain
        and mapped to their respective symbol slots."""

        # Build a basic communication frame containing only a single symbol slot with 100 data subcarriers
        num_elements = 50
        symbol_elements = [FrameElement(ElementType.DATA) for _ in range(num_elements)]
        resource = FrameResource(repetitions=2, cp_ratio=0.2, elements=symbol_elements)
        self.waveform_generator.add_resource(resource)

        section = FrameSymbolSection(pattern=[0], frame=self.waveform_generator)
        self.waveform_generator.add_section(section)

        # Generate two separate symbol streams, one modeling data, one modeling noise
        expected_data_symbols = np.exp(2j * self.random_generator.uniform(0, pi, 100))
        expected_noise_symbols = np.exp(2j * self.random_generator.uniform(0, pi, 100))

        # Mock modulation by directly calling the section modulator
        data_signal = section.modulate(expected_data_symbols)
        noise_signal = section.modulate(expected_noise_symbols)[..., np.newaxis]

        data_symbols, noise_symbols = section.demodulate(data_signal, noise_signal)

        assert_array_almost_equal(expected_data_symbols[..., np.newaxis], data_symbols)
        assert_array_almost_equal(expected_noise_symbols[..., np.newaxis, np.newaxis], noise_symbols)

    def test_frame_demodulation_symbol_transformation(self) -> None:
        """Data and noise signals should be properly transformed to frequency domain
        and mapped to their respective symbol slots."""

        # Build a basic communication frame containing only a single symbol slot with 100 data subcarriers
        num_elements = 50
        symbol_elements = [FrameElement(ElementType.DATA) for _ in range(num_elements)]
        resource = FrameResource(repetitions=2, cp_ratio=0.2, elements=symbol_elements)
        self.waveform_generator.add_resource(resource)

        section = FrameSymbolSection(pattern=[0], frame=self.waveform_generator)
        self.waveform_generator.add_section(section)

        # Generate two separate symbol streams, one modeling data, one modeling noise
        expected_data_symbols = np.exp(2j * self.random_generator.uniform(0, pi, 100))
        expected_noise_symbols = np.exp(2j * self.random_generator.uniform(0, pi, 100))

        # Mock modulation by directly calling the section modulator
        data_signal = section.modulate(expected_data_symbols)
        noise_signal = section.modulate(expected_noise_symbols)[..., np.newaxis]
        noise_variance = 0.

        data_symbols, noise_symbols, noise_variances = self.waveform_generator.demodulate(data_signal, noise_signal,
                                                                                          noise_variance)

        assert_array_almost_equal(expected_data_symbols, data_symbols)
        assert_array_almost_equal(expected_noise_symbols[..., np.newaxis], noise_symbols)

    def test_channel_propagation_impulse_response_assumptions(self) -> None:
        """Data and noise signals should be properly transformed to frequency domain
        and mapped to their respective symbol slots."""

        # Build a basic communication frame containing only a single symbol slot with 100 data subcarriers
        num_elements = 50
        symbol_elements = [FrameElement(ElementType.DATA) for _ in range(num_elements)]
        resource = FrameResource(repetitions=2, cp_ratio=0.2, elements=symbol_elements)
        self.waveform_generator.add_resource(resource)

        section = FrameSymbolSection(pattern=[0], frame=self.waveform_generator)
        self.waveform_generator.add_section(section)

        # Generate two separate symbol streams, one modeling data, one modeling noise
        expected_data_symbols = np.exp(2j * self.random_generator.uniform(0, pi, 100))
        expected_noise_symbols = np.exp(2j * self.random_generator.uniform(0, pi, 100))

        # Mock modulation by directly calling the section modulator
        data_signal = section.modulate(expected_data_symbols)
        noise_signal = section.modulate(expected_noise_symbols)
        noise_variance = 0.

        propagated_data_signal = data_signal * noise_signal

        data_symbols, impulse_symbols, noise_variances = self.waveform_generator.demodulate(propagated_data_signal,
                                                                                            noise_signal[
                                                                                                ..., np.newaxis],
                                                                                            noise_variance)

        equalized_data_symbols = data_symbols / impulse_symbols[:, 0]
        assert_array_almost_equal(equalized_data_symbols, expected_data_symbols)


    def test_reference_based_channel_estimation(self) -> None:
        """Reference-based channel estimation should properly estimate channel at reference points."""

        pass
