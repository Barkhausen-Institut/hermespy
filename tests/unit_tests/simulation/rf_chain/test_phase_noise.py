# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.simulation.rf_chain.phase_noise import NoPhaseNoise, OscillatorPhaseNoise
from hermespy.simulation import SimulatedDevice
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Egor Achkasov"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Egor Achjasov", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
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

    def test_yaml_serialization(self) -> None:
        """Test serialization to and from yaml"""

        test_yaml_roundtrip_serialization(self, self.pn)


class TestOscillatorPhaseNoise(TestCase):
    """Test the doi: 10.1109/TCSI.2013.2285698 phase noise implementation"""

    def setUp(self) -> None:
        self.K0 = 10 ** (-110 / 10)
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
        """Phase noise model should introduce the correct phase offset"""

        # generate signal
        # taken from _examples/library/getting_started.py
        operator = DuplexModem()
        operator.waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=0.9)
        operator.device = SimulatedDevice()
        transmission = operator.transmit()
        signal = transmission.signal
        noisy_signal = self.pn.add_noise(signal)

        num_samples = signal.num_samples
        sampling_rate = signal.sampling_rate

        pn_samples = self.pn._get_noise_samples(num_samples, 1, sampling_rate)

        # check if noised signal magnitude is the same as in the original signal
        clear_signal_avg_amp = np.average(np.abs(signal.samples), axis=1)
        noisy_signal_avg_amp = np.average(np.abs(noisy_signal.samples), axis=1)
        for i in range(signal.num_streams):
            np.testing.assert_approx_equal(clear_signal_avg_amp[i], noisy_signal_avg_amp[i])

        # check if arg(x′[n])−arg(pn[n])≈arg(x[n])
        arg_diffs = np.angle(noisy_signal.samples) - np.angle(pn_samples)
        arg_signal = np.angle(signal.samples)
        np.testing.assert_allclose(arg_diffs, arg_signal, atol=10e7)

        # check if the pn time domain starts close to zero
        assert np.all(np.abs(pn_samples[:, 0]) < 1e7)

    def test_yaml_serialization(self) -> None:
        """Test serialization to and from yaml"""

        test_yaml_roundtrip_serialization(self, self.pn)
