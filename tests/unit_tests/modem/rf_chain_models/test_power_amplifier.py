import unittest

import numpy as np
import numpy.testing as npt
from copy import deepcopy


from parameters_parser.parameters_rf_chain import ParametersRfChain
from modem.rf_chain_models.power_amplifier import PowerAmplifier
from matplotlib import pyplot as plt


class TestPowerAmplifier(unittest.TestCase):

    def setUp(self) -> None:

        self.tx_power = 1.0

        # create a power amplifier following Rapp's model (Input Backoff = 0dB)
        rapp_params = ParametersRfChain()
        rapp_params.power_amplifier = "RAPP"
        rapp_params.input_backoff_pa_db = 2
        rapp_params.rapp_smoothness_factor = 1
        self.pa_rapp = PowerAmplifier(rapp_params, self.tx_power)

        # create a power amplifier with ideal clipping (Input Backoff = 0dB)
        clip_params = ParametersRfChain()
        clip_params.power_amplifier = "CLIP"
        clip_params.input_backoff_pa_db = 0
        self.pa_clip = PowerAmplifier(clip_params, self.tx_power)

        # create a power amplifier with ideal clipping (Input Backoff = 3dB)
        clip_params_ibo3db = deepcopy(clip_params)
        clip_params_ibo3db.input_backoff_pa_db = 3

        self.pa_clip_ibo3db = PowerAmplifier(clip_params_ibo3db, self.tx_power)

        # create a power amplifier with Saleh's model (Input Backoff = 0dB)
        saleh_params = ParametersRfChain()
        saleh_params.power_amplifier = "SALEH"
        saleh_params.saleh_alpha_a = 1.9638
        saleh_params.saleh_alpha_phi = 2.5293
        saleh_params.saleh_beta_a = 0.9945
        saleh_params.saleh_beta_phi = 2.8168
        saleh_params.input_backoff_pa_db = 2
        self.pa_saleh = PowerAmplifier(saleh_params, self.tx_power)

        # create a custom power_amplifier
        custom_params = ParametersRfChain()
        custom_params.power_amplifier = "CUSTOM"
        custom_params.custom_pa_input = np.array([0.00881, 0.01109, 0.01396, 0.01757, 0.02213, 0.02786, 0.03508,
                                                  0.04416, 0.05559, 0.06998, 0.08810, 0.11092, 0.13964, 0.17579,
                                                  0.22131, 0.27861, 0.35075, 0.44157, 0.55590, 0.69984, 0.88105,
                                                  1.10917, 1.39637, 1.75792, 2.21309, 2.78612, 3.50752, 4.41570,
                                                  5.55904, 6.99842, 8.81049])
        custom_params.custom_pa_output = np.array([0.00881, 0.01109, 0.01396, 0.01757, 0.02213, 0.02786, 0.03503,
                                                   0.04411, 0.05553, 0.06990, 0.08800, 0.11078, 0.13932, 0.17539,
                                                   0.22055, 0.27701, 0.34754, 0.43501, 0.54263, 0.67143, 0.81564,
                                                   0.95830, 1.07771, 1.16815, 1.23595, 1.28529, 1.32434, 1.36301,
                                                   1.40120, 1.45211, 1.50661])
        custom_params.custom_pa_phase = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                                                  0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
                                                  0.19, .25, .30, .5, .8])
        custom_params.input_backoff_pa_db = 5
        custom_params._check_params()
        self.pa_custom = PowerAmplifier(custom_params, self.tx_power)

    def test_signal_properly_clipped(self) -> None:

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa_clip.send(signal)

        self.assertFalse(np.any(np.abs(output) > 1.0))

        non_distorted_index = np.abs(signal) <= 1.0
        np.testing.assert_array_equal(signal[non_distorted_index], output[non_distorted_index])

    def test_signal_properly_clipped_with_backoff (self) -> None:

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa_clip_ibo3db.send(signal)

        backoff = 10**(self.pa_clip_ibo3db.params.input_backoff_pa_db / 20)

        self.assertAlmostEqual(np.max(np.abs(output)), backoff)

        non_distorted_index = np.abs(signal) <= backoff
        np.testing.assert_array_equal(signal[non_distorted_index], output[non_distorted_index])

    def test_signal_properly_distorted_with_rapp(self) -> None:

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa_rapp.send(signal)

        rms_voltage = np.sqrt(self.tx_power)
        saturation_amp = rms_voltage * 10**(self.pa_rapp.params.input_backoff_pa_db / 20.)

        p = self.pa_rapp.params.rapp_smoothness_factor
        expected_output = signal / (1 + (np.abs(signal) / self.pa_rapp.saturation_amplitude)**(2 * p)) ** (1 / (2 * p))

        np.testing.assert_allclose(output, expected_output)

    def test_signal_properly_distorted_with_saleh(self) -> None:

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)
        output = self.pa_saleh.send(signal)

        rms_voltage = np.sqrt(self.tx_power)
        saturation_amp = rms_voltage * 10**(self.pa_saleh.params.input_backoff_pa_db / 20.)

        r = np.abs(signal) / saturation_amp

        expected_amp = (r * self.pa_saleh.params.saleh_alpha_a / (1 + self.pa_saleh.params.saleh_beta_a * r**2) *
                        saturation_amp)
        expected_phase = (np.angle(signal) +
                          (r**2 * self.pa_saleh.params.saleh_alpha_phi /
                           (1 + self.pa_saleh.params.saleh_beta_phi * r**2)))

        expected_output = expected_amp * np.exp(1j * expected_phase)

        np.testing.assert_allclose(output, expected_output)

    def test_signal_properly_distorted_with_custom(self) -> None:

        number_of_samples = 100

        rnd_indices = np.random.randint(self.pa_custom.params.custom_pa_input.size, size=number_of_samples)

        rms_voltage = np.sqrt(self.tx_power)
        saturation_amp = rms_voltage * 10**(self.pa_custom.params.input_backoff_pa_db / 20.)

        phases = 2 * np.pi * np.random.random(number_of_samples)
        signal = (self.pa_custom.params.custom_pa_input[rnd_indices]
                  * (saturation_amp / rms_voltage)) * np.exp(1j * phases)

        output = self.pa_custom.send(signal)

        expected_amp = self.pa_custom.params.custom_pa_output[rnd_indices] * (saturation_amp / rms_voltage)
        expected_phase = phases + self.pa_custom.params.custom_pa_phase[rnd_indices]

        expected_output = expected_amp * np.exp(1j * expected_phase)

        np.testing.assert_allclose(output, expected_output)

    def test_power_adjusted(self) -> None:

        number_of_samples = 1000
        signal = np.random.normal(size=number_of_samples) + 1j*np.random.normal(size=number_of_samples)

        self.pa_rapp.params.adjust_power_after_pa = True
        output = self.pa_rapp.send(signal)
        self.pa_rapp.params.adjust_power_after_pa = False

        power_in = np.mean(np.real(signal)**2 + np.imag(signal)**2)
        power_out = np.mean(np.real(output)**2 + np.imag(output)**2)

        self.assertAlmostEqual(power_in, power_out)

    def plot_pa_responses(self) -> None:
        """
        Plots the AM/AM response of the clipping and Rapp amplifier models for visual inspection
        """
        pa_rapp_pa10 = deepcopy(self.pa_rapp)
        pa_rapp_pa10.params.rapp_smoothness_factor = 10

        input_signal = np.linspace(0, 2.0, 201) + 0j
        output_pa1 = (self.pa_rapp.send(input_signal * self.pa_rapp.saturation_amplitude) /
                      self.pa_rapp.saturation_amplitude)
        output_pa10 = (pa_rapp_pa10.send(input_signal * pa_rapp_pa10.saturation_amplitude) /
                       pa_rapp_pa10.saturation_amplitude)
        output_clip = (self.pa_clip.send(input_signal * self.pa_clip.saturation_amplitude) /
                       self.pa_clip.saturation_amplitude)
        output_saleh = (self.pa_saleh.send(input_signal * self.pa_saleh.saturation_amplitude) /
                        self.pa_saleh.saturation_amplitude)
        output_custom = (self.pa_custom.send(input_signal * self.pa_custom.saturation_amplitude) /
                         self.pa_custom.saturation_amplitude)

        plt.plot(input_signal, output_pa1, label="RAPP, p=1")
        plt.plot(input_signal, output_pa10, label="RAPP, p=10")
        plt.plot(input_signal, output_clip, label="CLIP")
        plt.plot(input_signal, np.abs(output_saleh), label="SALEH")
        plt.plot(input_signal, np.abs(output_custom), label="CUSTOM")

        plt.legend()
        plt.xlabel("Input Amplitude")
        plt.xlim([0, 2])
        plt.ylabel("Output Amplitude")
        plt.ylim([0, 1.4])
        plt.title("AM/AM Response")
        plt.grid()


if __name__ == '__main__':

    test = TestPowerAmplifier()
    test.setUp()
    test.plot_pa_responses()

    plt.show()

    unittest.main()
