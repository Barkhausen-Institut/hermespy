# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from hermespy.core import Signal
from hermespy.simulation.rf_chain import RfChain

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRfChain(TestCase):
    """Test RF chain model"""

    def setUp(self) -> None:
        self.phase_offset = 1.0
        self.amplitude_imbalance = 1e-3

        self.rf_chain = RfChain(self.phase_offset, self.amplitude_imbalance)

    def test_initiation(self) -> None:
        """Initialization arguments should be properly stored"""

        self.assertEqual(self.rf_chain.phase_offset, self.phase_offset)
        self.assertEqual(self.rf_chain.amplitude_imbalance, self.amplitude_imbalance)

    def test_amplitude_imbalance_validation(self) -> None:
        """Amplitude imbalance property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.rf_chain.amplitude_imbalance = 1.1

    def test_amplitude_imbalance_setget(self) -> None:
        """Amplitude imbalance property getter should return setter argument"""

        expected_imbalance = 0.5
        self.rf_chain.amplitude_imbalance = expected_imbalance

        self.assertEqual(self.rf_chain.amplitude_imbalance, expected_imbalance)

    def test_phase_offset_setget(self) -> None:
        """Phase offset property getter should return setter argument"""

        expected_offset = 0.5
        self.rf_chain.phase_offset = expected_offset

        self.assertEqual(self.rf_chain.phase_offset, expected_offset)

    def test_power_amplifier_setget(self) -> None:
        """Power amplifier property getter should return setter argument"""

        expected_pa = Mock()
        self.rf_chain.power_amplifier = expected_pa

        self.assertIs(expected_pa, self.rf_chain.power_amplifier)

    def test_phase_noise_setget(self) -> None:
        """Phase noise property getter should return setter argument"""

        expected_noise = Mock()
        self.rf_chain.phase_noise = expected_noise

        self.assertIs(expected_noise, self.rf_chain.phase_noise)

    def test_transmit_power_amplifier_integration(self) -> None:
        """Power amplifier should be called during transmit"""

        signal = Signal.Empty(1.0, 1, 0, carrier_frequency=0.0)

        pa = Mock()
        pa.send.side_effect = lambda x: x
        self.rf_chain.power_amplifier = pa

        self.rf_chain.transmit(signal)

        pa.send.assert_called_once()
