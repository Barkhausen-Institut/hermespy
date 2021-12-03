# -*- coding: utf-8 -*-
"""Test Power Amplifier Models."""

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from hermespy.modem.rf_chain_models.power_amplifier import PowerAmplifier

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRappPowerAmplifier(unittest.TestCase):
    """Test the Rapp power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.tx_power = 1.0
        self.input_backoff_pa_db = 2
        self.rapp_smoothness_factor = 1

        self.pa = PowerAmplifier(PowerAmplifier.Model.RAPP,
                                 tx_power=self.tx_power,
                                 input_backoff_pa_db=self.input_backoff_pa_db,
                                 rapp_smoothness_factor=self.rapp_smoothness_factor)

    def test_signal_distortion(self) -> None:
        """Signal should be properly distorted."""

        number_of_samples = 1000
        signal = self.rng.normal(size=number_of_samples) + 1j*self.rng.normal(size=number_of_samples)
        output = self.pa.send(signal)

        rms_voltage = np.sqrt(self.tx_power)
        saturation_amp = rms_voltage * 10**(self.input_backoff_pa_db / 20.)

        p = self.rapp_smoothness_factor
        expected_output = signal / (1 + (np.abs(signal) / saturation_amp)**(2 * p)) ** (1 / (2 * p))

        assert_array_almost_equal(expected_output, output)

    def test_power_adjusted(self) -> None:

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)

        self.pa.adjust_power_after_pa = True
        output = self.pa.send(signal)

        power_in = np.mean(np.real(signal)**2 + np.imag(signal)**2)
        power_out = np.mean(np.real(output)**2 + np.imag(output)**2)

        self.assertAlmostEqual(power_in, power_out)


class TestClipPowerAmplifier(unittest.TestCase):
    """Test the Clip power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.tx_power = 1.0
        self.input_backoff_pa_db = 0.

        self.pa = PowerAmplifier(PowerAmplifier.Model.CLIP,
                                 tx_power=self.tx_power,
                                 input_backoff_pa_db=self.input_backoff_pa_db)

    def test_signal_clipped(self) -> None:
        """Signal should be properly clipped"""

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa.send(signal)

        self.assertFalse(np.any(np.abs(output) > 1.0))

        non_distorted_index = np.abs(signal) <= 1.0
        assert_array_equal(signal[non_distorted_index], output[non_distorted_index])

    def test_signal_clipped_backoff(self) -> None:

        input_backoff_pa_db = 3
        self.pa = PowerAmplifier(PowerAmplifier.Model.CLIP,
                                 tx_power=self.tx_power,
                                 input_backoff_pa_db=input_backoff_pa_db)

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa.send(signal)

        backoff = 10**(input_backoff_pa_db / 20)

        self.assertAlmostEqual(np.max(np.abs(output)), backoff)

        non_distorted_index = np.abs(signal) <= backoff
        np.testing.assert_array_equal(signal[non_distorted_index], output[non_distorted_index])


class TestSalehPowerAmplifier(unittest.TestCase):
    """Test the Saleh power amplifier model."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.tx_power = 1.0
        self.alpha_a = 1.9638
        self.alpha_phi = 2.5293
        self.beta_a = 0.9945
        self.beta_phi = 2.8168
        self.input_backoff_pa_db = 2

        self.pa = PowerAmplifier(PowerAmplifier.Model.SALEH,
                                 tx_power=self.tx_power,
                                 input_backoff_pa_db=self.input_backoff_pa_db,
                                 saleh_alpha_a=self.alpha_a,
                                 saleh_alpha_phi=self.alpha_phi,
                                 saleh_beta_a=self.beta_a,
                                 saleh_beta_phi=self.beta_phi)

    def test_signal_distortion(self) -> None:
        """Signal should be properly distorted."""

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa.send(signal)

        rms_voltage = np.sqrt(self.tx_power)
        saturation_amp = rms_voltage * 10**(self.input_backoff_pa_db / 20.)

        r = np.abs(signal) / saturation_amp

        expected_amp = (r * self.alpha_a / (1 + self.beta_a * r**2) *
                        saturation_amp)
        expected_phase = (np.angle(signal) +
                          (r**2 * self.alpha_phi /
                           (1 + self.beta_phi * r**2)))

        expected_output = expected_amp * np.exp(1j * expected_phase)
        assert_array_almost_equal(output, expected_output)


class TestCustomPowerAmplifier(unittest.TestCase):
    """Test the custom power amplifier model."""

    def setUp(self) -> None:

        self.tx_power = 1.
        self.pa_input = np.array([0.00881, 0.01109, 0.01396, 0.01757, 0.02213, 0.02786, 0.03508,
                                  0.04416, 0.05559, 0.06998, 0.08810, 0.11092, 0.13964, 0.17579,
                                  0.22131, 0.27861, 0.35075, 0.44157, 0.55590, 0.69984, 0.88105,
                                  1.10917, 1.39637, 1.75792, 2.21309, 2.78612, 3.50752, 4.41570,
                                  5.55904, 6.99842, 8.81049])
        self.pa_output = np.array([0.00881, 0.01109, 0.01396, 0.01757, 0.02213, 0.02786, 0.03503,
                                   0.04411, 0.05553, 0.06990, 0.08800, 0.11078, 0.13932, 0.17539,
                                   0.22055, 0.27701, 0.34754, 0.43501, 0.54263, 0.67143, 0.81564,
                                   0.95830, 1.07771, 1.16815, 1.23595, 1.28529, 1.32434, 1.36301,
                                   1.40120, 1.45211, 1.50661])
        self.pa_phase = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                                  0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
                                  0.19, .25, .30, .5, .8])
        self.pa_gain = self.pa_output / self.pa_input

        self.input_backoff_pa_db = 5

        self.pa = PowerAmplifier(PowerAmplifier.Model.CUSTOM, self.tx_power,
                                 custom_pa_input=self.pa_input,
                                 custom_pa_output=self.pa_output,
                                 custom_pa_phase=self.pa_phase,
                                 custom_pa_gain=self.pa_gain,
                                 input_backoff_pa_db=self.input_backoff_pa_db)

    def test_signal_distortion(self) -> None:
        """Signal should be properly distorted."""

        number_of_samples = 100

        rnd_indices = np.random.randint(self.pa_input.size, size=number_of_samples)

        rms_voltage = np.sqrt(self.tx_power)
        saturation_amp = rms_voltage * 10**(self.input_backoff_pa_db / 20.)

        phases = 2 * np.pi * np.random.random(number_of_samples)
        signal = (self.pa_input[rnd_indices]
                  * (saturation_amp / rms_voltage)) * np.exp(1j * phases)

        output = self.pa.send(signal)

        expected_amp = self.pa_output[rnd_indices] * (saturation_amp / rms_voltage)
        expected_phase = phases + self.pa_phase[rnd_indices]

        expected_output = expected_amp * np.exp(1j * expected_phase)
        assert_array_almost_equal(expected_output, output)
