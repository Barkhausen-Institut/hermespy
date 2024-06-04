# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from hermespy.modem import BaseModem, CommunicationWaveform
from hermespy.simulation.modem.noise import CommunicationNoiseLevel, EBN0, ESN0

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockCommunicationNoiseLevel(CommunicationNoiseLevel):
    """Implementation of the communication noise level base class for testing purposes."""

    def get_power(self) -> float:
        return 0.123

    @property
    def title(self) -> str:
        return "Mock"


class TestCommunicationNoiseLevel(TestCase):
    """Test the communication noise level base class"""

    def setUp(self) -> None:

        self.reference = Mock()
        self.level = 1.234
        self.noise_level = MockCommunicationNoiseLevel(self.reference, self.level)

    def test_init(self) -> None:
        """Initialization parameters should be stored correctly"""

        self.assertIs(self.reference, self.noise_level.reference)
        self.assertEqual(self.level, self.noise_level.level)

    def test_level_setget(self) -> None:
        """Level property getter should return setter argument"""

        expected_level = 1.234
        self.noise_level.level = expected_level
        self.assertEqual(expected_level, self.noise_level.level)

    def test_level_validation(self) -> None:
        """Level property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.noise_level.level = -1.0

    def test_reference_setget(self) -> None:
        """Reference property getter should return setter argument"""

        expected_reference = Mock()
        self.noise_level.reference = expected_reference
        self.assertEqual(expected_reference, self.noise_level.reference)

    def test_get_reference_waveform_validation(self) -> None:
        """Get reference subroutine should raise ValueError for invalid reference"""

        reference = Mock(spec=BaseModem)
        reference.waveform = None
        self.noise_level.reference = reference

        with self.assertRaises(RuntimeError):
            _ = self.noise_level._get_reference_waveform()

    def test_get_reference_waveform(self) -> None:
        """Get reference subroutine should return the waveform of the reference"""

        reference_waveform = Mock(spec=CommunicationWaveform)
        self.noise_level.reference = reference_waveform
        self.assertIs(reference_waveform, self.noise_level._get_reference_waveform())

        reference_modem = Mock(spec=BaseModem)
        reference_modem.waveform = reference_waveform
        self.noise_level.reference = reference_modem
        self.assertIs(reference_waveform, self.noise_level._get_reference_waveform())


class TestEBN0(TestCase):
    """Test the EBN0 communication noise level class"""

    def setUp(self) -> None:

        self.reference = Mock(spec=CommunicationWaveform)
        self.level = 1.234
        self.ebn0 = EBN0(self.reference, self.level)

    def test_title(self) -> None:
        """Title property should return the title of the class"""

        self.assertIsInstance(self.ebn0.title, str)

    def test_get_power(self) -> None:
        """Power should be the bit energy divided by the noise level"""

        bit_energy = 0.123
        self.reference.bit_energy = 0.123
        expected_power = bit_energy / self.level

        self.assertEqual(expected_power, self.ebn0.get_power())


class TestESN0(TestCase):
    """Test the ESN0 communication noise level class"""

    def setUp(self) -> None:

        self.reference = Mock(spec=CommunicationWaveform)
        self.level = 1.234
        self.esn0 = ESN0(self.reference, self.level)

    def test_title(self) -> None:
        """Title property should return the title of the class"""

        self.assertIsInstance(self.esn0.title, str)

    def test_get_power(self) -> None:
        """Power should be the symbol energy divided by the noise level"""

        symbol_energy = 0.123
        self.reference.symbol_energy = 0.123
        expected_power = symbol_energy / self.level

        self.assertEqual(expected_power, self.esn0.get_power())
