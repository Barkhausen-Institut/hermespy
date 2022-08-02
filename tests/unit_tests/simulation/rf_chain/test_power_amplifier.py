# -*- coding: utf-8 -*-
"""Test Power Amplifier Models."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_array_less

from hermespy.simulation.rf_chain.power_amplifier import \
    PowerAmplifier, ClippingPowerAmplifier, RappPowerAmplifier, SalehPowerAmplifier, CustomPowerAmplifier

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPowerAmplifier(unittest.TestCase):
    """Test power amplifier model base class."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_samples = 1000

        self.saturation_amplitude = .5
        self.pa = PowerAmplifier(saturation_amplitude=self.saturation_amplitude)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""

        self.assertEqual(self.saturation_amplitude, self.pa.saturation_amplitude)

    def test_saturation_amplitude_setget(self) -> None:
        """Saturation amplitude property getter should return setter argument."""

        saturation_amplitude = 100
        self.pa.saturation_amplitude = saturation_amplitude

        self.assertEqual(saturation_amplitude, self.pa.saturation_amplitude)

    def test_saturation_amplitude_validation(self) -> None:
        """Saturation amplitude property setter should raise ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.pa.saturation_amplitude = -1.0

        try:
            self.pa.saturation_amplitude = 0.

        except ValueError:
            self.fail()

    def test_adjust_power_setget(self) -> None:
        """Adjust power flag get should return set value."""

        self.pa.adjust_power = True
        self.assertEqual(True, self.pa.adjust_power)

    def test_send_no_adjust(self) -> None:
        """Sending a signal without adjustment should not alter the signal in any way."""

        expected_signal = self.rng.normal(size=self.num_samples) + 1j*self.rng.normal(size=self.num_samples)
        signal = self.pa.send(expected_signal)

        assert_array_equal(expected_signal, signal)

    def test_send_adjust(self) -> None:
        """Power loss should be adjusted if the respective flag is enabled."""

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)

        with patch.object(self.pa, 'model', new=lambda input_signal: 2 * input_signal):

            self.pa.adjust_power = True
            output = self.pa.send(signal)

        power_in = np.mean(np.real(signal)**2 + np.imag(signal)**2)
        power_out = np.mean(np.real(output)**2 + np.imag(output)**2)

        self.assertAlmostEqual(power_in, power_out)

    def test_model(self) -> None:
        """Model function should be a stub directly returning the input as output."""

        expected_signal = self.rng.normal(size=self.num_samples) + 1j*self.rng.normal(size=self.num_samples)
        signal = self.pa.send(expected_signal)

        assert_array_equal(expected_signal, signal)


class TestRappPowerAmplifier(unittest.TestCase):
    """Test the Rapp power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_samples = 1000

        self.saturation_amplitude = 10 ** (2 / 10)
        self.smoothness_factor = 1.

        self.pa = RappPowerAmplifier(saturation_amplitude=self.saturation_amplitude,
                                     smoothness_factor=self.smoothness_factor)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""

        self.assertEqual(self.saturation_amplitude, self.pa.saturation_amplitude)
        self.assertEqual(self.smoothness_factor, self.pa.smoothness_factor)

    def test_smoothness_factor_setget(self) -> None:
        """Smoothness factor property getter should return setter argument."""

        smoothness_factor = 1.23
        self.pa.smoothness_factor = 1.23

        self.assertEqual(smoothness_factor, self.pa.smoothness_factor)

    def test_smoothness_factor_validation(self) -> None:
        """Smoothness factor property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.pa.smoothness_factor = 0.
            
        with self.assertRaises(ValueError):
            self.pa.smoothness_factor = -1.

        try:
            self.pa.smoothness_factor = 1.0

        except ValueError:
            self.fail()

    def test_model(self) -> None:
        """Signal should be properly distorted."""

        signal = self.rng.normal(size=self.num_samples) + 1j*self.rng.normal(size=self.num_samples)
        output = self.pa.model(signal)

        p = self.smoothness_factor
        expected_output = signal / (1 + (np.abs(signal) / self.saturation_amplitude)**(2 * p)) ** (1 / (2 * p))

        assert_array_almost_equal(expected_output, output)


class TestClippingPowerAmplifier(unittest.TestCase):
    """Test the Clipping power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_samples = 1000

        self.saturation_amplitude = 10 ** (2 / 10)

        self.pa = ClippingPowerAmplifier(saturation_amplitude=self.saturation_amplitude)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""

        self.assertEqual(self.saturation_amplitude, self.pa.saturation_amplitude)

    def test_model(self) -> None:
        """Signal should be properly clipped."""

        signal = np.random.normal(size=self.num_samples) + 1j*np.random.normal(size=self.num_samples)
        output = self.pa.model(signal)

        assert_array_less(output, self.saturation_amplitude + 1e-5)

        non_distorted_index = np.abs(signal) <= 1.0
        assert_array_equal(signal[non_distorted_index], output[non_distorted_index])


class TestSalehPowerAmplifier(unittest.TestCase):
    """Test the Saleh power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_samples = 1000

        self.saturation_amplitude = 10 ** (2 / 10)
        self.amplitude_alpha = 1.9638
        self.amplitude_beta = 0.9945
        self.phase_alpha = 2.5293
        self.phase_beta = 2.8168

        self.pa = SalehPowerAmplifier(saturation_amplitude=self.saturation_amplitude,
                                      amplitude_alpha=self.amplitude_alpha,
                                      amplitude_beta=self.amplitude_beta,
                                      phase_alpha=self.phase_alpha,
                                      phase_beta=self.phase_beta)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""

        self.assertEqual(self.saturation_amplitude, self.pa.saturation_amplitude)
        self.assertEqual(self.amplitude_alpha, self.pa.amplitude_alpha)
        self.assertEqual(self.amplitude_beta, self.pa.amplitude_beta)
        self.assertEqual(self.phase_alpha, self.pa.phase_alpha)
        self.assertEqual(self.phase_beta, self.pa.phase_beta)
        
    def test_amplitude_alpha_setget(self) -> None:
        """Amplitude alpha property getter should return setter argument."""

        amplitude_alpha = 1.23
        self.pa.amplitude_alpha = 1.23

        self.assertEqual(amplitude_alpha, self.pa.amplitude_alpha)

    def test_amplitude_alpha_validation(self) -> None:
        """Amplitude alpha property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.pa.amplitude_alpha = -1.0

        try:
            self.pa.amplitude_alpha = 0.0

        except ValueError:
            self.fail()
            
    def test_amplitude_beta_setget(self) -> None:
        """Amplitude beta property getter should return setter argument."""

        amplitude_beta = 1.23
        self.pa.amplitude_beta = 1.23

        self.assertEqual(amplitude_beta, self.pa.amplitude_beta)

    def test_amplitude_beta_validation(self) -> None:
        """Amplitude beta property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.pa.amplitude_beta = -1.0

        try:
            self.pa.amplitude_beta = 0.0

        except ValueError:
            self.fail()
            
    def test_phase_alpha_setget(self) -> None:
        """Phase alpha property getter should return setter argument."""

        phase_alpha = 1.23
        self.pa.phase_alpha = 1.23

        self.assertEqual(phase_alpha, self.pa.phase_alpha)
        
    def test_phase_beta_setget(self) -> None:
        """Phase beta property getter should return setter argument."""

        phase_beta = 1.23
        self.pa.phase_beta = 1.23

        self.assertEqual(phase_beta, self.pa.phase_beta)

    def test_model(self) -> None:
        """Signal should be properly distorted."""

        self.rng = np.random.default_rng(42)
        self.num_samples = 1000

        signal = np.random.normal(size=self.num_samples) + 1j*np.random.normal(size=self.num_samples)
        output = self.pa.send(signal)

        r = np.abs(signal) / self.saturation_amplitude

        expected_amp = (r * self.amplitude_alpha / (1 + self.amplitude_beta * r**2) * self.saturation_amplitude)
        expected_phase = (np.angle(signal) + (r**2 * self.phase_alpha / (1 + self.phase_beta * r**2)))

        expected_output = expected_amp * np.exp(1j * expected_phase)
        assert_array_almost_equal(output, expected_output)


class TestCustomPowerAmplifier(unittest.TestCase):
    """Test the custom power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_samples = 1000

        self.saturation_amplitude = 10 ** (5 / 10)
        self.input = np.array([0.00881, 0.01109, 0.01396, 0.01757, 0.02213, 0.02786, 0.03508,
                               0.04416, 0.05559, 0.06998, 0.08810, 0.11092, 0.13964, 0.17579,
                               0.22131, 0.27861, 0.35075, 0.44157, 0.55590, 0.69984, 0.88105,
                               1.10917, 1.39637, 1.75792, 2.21309, 2.78612, 3.50752, 4.41570,
                               5.55904, 6.99842, 8.81049])
        self.output = np.array([0.00881, 0.01109, 0.01396, 0.01757, 0.02213, 0.02786, 0.03503,
                                0.04411, 0.05553, 0.06990, 0.08800, 0.11078, 0.13932, 0.17539,
                                0.22055, 0.27701, 0.34754, 0.43501, 0.54263, 0.67143, 0.81564,
                                0.95830, 1.07771, 1.16815, 1.23595, 1.28529, 1.32434, 1.36301,
                                1.40120, 1.45211, 1.50661])
        self.phase = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                               0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
                               0.19, .25, .30, .5, .8])
        self.gain = self.output / self.input

        self.pa = CustomPowerAmplifier(saturation_amplitude=self.saturation_amplitude,
                                       input=self.input,
                                       gain=self.gain,
                                       phase=self.phase)

    def test_model(self) -> None:
        """Signal should be properly distorted."""

        rnd_indices = self.rng.integers(self.input.size, size=self.num_samples)
        phases = 2 * np.pi * self.rng.random(self.num_samples)
        signal = self.input[rnd_indices] * self.saturation_amplitude * np.exp(1j * phases)

        output = self.pa.model(signal)

        expected_amp = self.output[rnd_indices] * self.saturation_amplitude
        expected_phase = phases + self.phase[rnd_indices]

        expected_output = expected_amp * np.exp(1j * expected_phase)
        assert_array_almost_equal(expected_output, output)
