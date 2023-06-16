# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from matplotlib import pyplot as plt

from hermespy.core import Signal
from hermespy.simulation.rf_chain.phase_noise import NoPhaseNoise, OscillatorPhaseNoise
from hermespy.simulation import SimulatedDevice
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNoPhaseNoise(TestCase):
    """Test the phase noise stub"""

    def setUp(self) -> None:

        self.pn = NoPhaseNoise()

    def test_add_noise(self) -> None:
        """Adding noise should actually do nothing"""

        signal = Signal(np.random.standard_normal((3, 10)), 1)
        noisy_signal = self.pn.add_noise(signal)

        assert_array_equal(signal.samples, noisy_signal.samples)


class TestOscillatorPhaseNoise(TestCase):
    """Test the doi: 10.1109/TCSI.2013.2285698 phase noise implementation"""

    def setUp(self) -> None:
        self.K0 = 10**(-110 / 10)
        self.K2 = 10
        self.K3 = 10**4
        self.pn = OscillatorPhaseNoise(self.K0, self.K2, self.K3)
        self.pn0 = OscillatorPhaseNoise(self.K0, 0, 0)
        self.pn2 = OscillatorPhaseNoise(0, self.K2, 0)
        self.pn3 = OscillatorPhaseNoise(0, 0, self.K3)

    def test_K0_validation(self) -> None:
        """K0 property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.pn.K0 = -0.1

    def test_K0_setget(self) -> None:
        """K0 property getter should return setter argument"""
        self.pn.K0 = self.K0

        self.assertEqual(self.pn.K0, self.K0)

    def test_K2_validation(self) -> None:
        """K2 property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.pn.K2 = -0.1

    def test_K2_setget(self) -> None:
        """K2 property getter should return setter argument"""
        self.pn.K2 = self.K2

        self.assertEqual(self.pn.K2, self.K2)

    def test_K3_validation(self) -> None:
        """K3 property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.pn.K3 = -0.1

    def test_K3_setget(self) -> None:
        """K3 property getter should return setter argument"""
        self.pn.K3 = self.K3

        self.assertEqual(self.pn.K3, self.K3)

    def test_add_noise(self) -> None:
        # generate signal
        # taken from _examples/library/getting_started.py
        operator = DuplexModem()
        operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
        operator.device = SimulatedDevice()
        transmission = operator.transmit()
        signal = transmission.signal
        noisy_signal = self.pn.add_noise(signal)

        # plot singular K noise PSDs
        # ==========================================================================================
        fig1, axes1 = plt.subplots(4, 1)

        num_samples = signal.num_samples
        sampling_rate = signal.sampling_rate

        pn0_samples = self.pn0._get_noise_samples(num_samples, 1, sampling_rate)
        pn2_samples = self.pn2._get_noise_samples(num_samples, 1, sampling_rate)
        pn3_samples = self.pn3._get_noise_samples(num_samples, 1, sampling_rate)
        pn_samples = self.pn._get_noise_samples(num_samples, 1, sampling_rate)

        sampling_interval = 1 / sampling_rate
        var_w0 = self.K0 / sampling_interval
        var_w2 = 4 * self.K2 * sampling_interval * np.pi ** 2
        var_w3 = 8 * self.K3 * sampling_interval ** 2 * np.pi ** 3

        _, freqs0 = axes1[0].psd(pn0_samples[0], NFFT=num_samples, pad_to=2**18, Fs=sampling_rate,
                                 sides='onesided', scale_by_freq=True, color='green',
                                 label='K0 PN PSD')
        axes1[0].plot(freqs0, 10 * np.log10(np.ones_like(freqs0)*self.K0), color='blue',
                      linestyle='dotted', label='K0')
        axes1[0].plot(freqs0, 10 * np.log10(np.ones_like(freqs0)*self.K0) + 10 * np.log10(3 * np.sqrt(var_w0)),
                      color='blue', label='K0 + s')
        _, freqs2 = axes1[1].psd(pn2_samples[0], NFFT=num_samples, pad_to=2**18, Fs=sampling_rate,
                                 sides='onesided', scale_by_freq=True, color='green',
                                 label='K2 PN PSD')
        axes1[1].plot(freqs2, 10 * np.log10(self.K2 / freqs2 ** 2), color='black', linestyle='dashed',
                      label='K2 / f^2')
        axes1[1].plot(freqs2, 10 * np.log10(self.K2 / freqs2 ** 2) + 10 * np.log10(3 * np.sqrt(var_w2)),
                      color='black', label='K2 + s')
        _, freqs3 = axes1[2].psd(pn3_samples[0], NFFT=num_samples, pad_to=2**18, Fs=sampling_rate,
                                 sides='onesided', scale_by_freq=True, color='green',
                                 label='K3 PN PSD')
        axes1[2].plot(freqs3, 10 * np.log10(self.K3 / freqs3 ** 3), color='red', linestyle='solid',
                      label='K3 / f^3')
        axes1[2].plot(freqs3, 10 * np.log10(self.K3 / freqs3 ** 3) + 10 * np.log10(3 * np.sqrt(var_w3)),
                      color='red', label='K3 + s')
        _, freqs = axes1[3].psd(pn_samples[0], NFFT=num_samples, pad_to=2**18, Fs=sampling_rate,
                                sides='onesided', scale_by_freq=True, color='green', label='PN PSD')
        axes1[3].plot(freqs, 10 * np.log10(np.ones_like(freqs)*self.K0), color='blue',
                      linestyle='dotted', label='K0')
        axes1[3].plot(freqs, 10 * np.log10(self.K2 / freqs ** 2), color='black', linestyle='dashed',
                      label='K2 / f^2')
        axes1[3].plot(freqs, 10 * np.log10(self.K3 / freqs ** 3), color='red', linestyle='solid',
                      label='K3 / f^3')

        for i in range(axes1.shape[0]):
            axes1[i].set_xscale('log')
            axes1[i].set_xlim([50, 10**6])
            axes1[i].set_ylim([-120, 0])
            axes1[i].legend()
        fig1.show()

        # ==========================================================================================
        # DONE plot singular K noise PSDs

        # assert generated PN components variance
        # ================================================

        # TODO

        # ===============================================
        # DONE assert generated PN components variance

        # test noised signal magnitude, angle deviations and initial pn values
        # ===============================================

        # check if noised signal magnitude is the same as in the original signal
        clear_signal_avg_amp = np.average(np.abs(signal.samples), axis=1)
        noisy_signal_avg_amp = np.average(np.abs(noisy_signal.samples), axis=1)
        for i in range(signal.num_streams):
            np.testing.assert_approx_equal(clear_signal_avg_amp[i],
                                           noisy_signal_avg_amp[i])

        # check if arg(x′[n])−arg(pn[n])≈arg(x[n])
        arg_diffs = np.angle(noisy_signal.samples) - np.angle(pn_samples)
        arg_signal = np.angle(signal.samples)
        np.testing.assert_allclose(arg_diffs, arg_signal, atol=10e7)

        # check if the pn time domain starts close to zero
        assert (np.all(np.abs(pn_samples[:, 0]) < 1e7))

        # ===============================================
        # DONE test noised signal magnitude, angle deviations and initial pn values


if __name__ == '__main__':
    unittest.main()
