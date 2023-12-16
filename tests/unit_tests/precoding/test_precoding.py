# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import DuplexModem, ReceivingModem
from hermespy.precoding import Precoder, Precoding
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PrecoderMock(Precoder):
    "Precoder mock for testing purposes"

    @property
    def num_input_streams(self) -> int:
        return 1

    @property
    def num_output_streams(self) -> int:
        return 1


class PrecodingMock(Precoding[PrecoderMock]):
    "Precoding mock for testing purposes"
    ...  # pragma: no cover


class TestPrecoder(TestCase):
    """Test base class for all precoders"""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.modem = DuplexModem()
        self.modem.device = self.device

        self.precoding = PrecodingMock(self.modem)
        self.precoder = PrecoderMock()
        self.precoding[0] = self.precoder

    def test_precoding_setget(self) -> None:
        """Precoding property getter should return setter argument"""

        expected_precoding = PrecodingMock()
        self.precoder.precoding = expected_precoding

        self.assertIs(self.precoder.precoding, expected_precoding)

    def test_required_num_input_streams_validation(self) -> None:
        """Querying the required number of input streams should raise an error if precoding is not set"""

        self.precoder.precoding = None
        with self.assertRaises(RuntimeError):
            _ = self.precoder.required_num_input_streams

    def test_required_num_input_streams(self) -> None:
        """Required number of input streams should report the correct value"""

        self.assertEqual(1, self.precoder.required_num_input_streams)

    def test_required_num_output_streams_validation(self) -> None:
        """Querying the required number of output streams should raise an error if precoding is not set"""

        self.precoder.precoding = None
        with self.assertRaises(RuntimeError):
            _ = self.precoder.required_num_output_streams

    def test_required_num_output_streams(self) -> None:
        """Required number of output streams should report the correct value"""

        self.assertEqual(1, self.precoder.required_num_output_streams)

    def test_rate(self) -> None:
        """Rate should report the correct value"""

        self.assertEqual(1, self.precoder.rate)


class TestPrecoding(TestCase):
    """Test base class for all precodings"""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.modem = DuplexModem()
        self.modem.device = self.device

        self.precoding = PrecodingMock(self.modem)
        self.precoder = PrecoderMock()
        self.precoding[0] = self.precoder

    def test_modem_setget(self) -> None:
        """Modem property getter should return setter argument"""

        expected_modem = DuplexModem()
        self.precoding.modem = expected_modem

        self.assertIs(self.precoding.modem, expected_modem)

    def test_required_outputs_validation(self) -> None:
        """Querying the required number of output streams should raise an error if precoding is not set"""

class TestPrecoding(TestCase):
    """Test base class for all precodings"""
    
    def setUp(self) -> None:
        
        self.device = SimulatedDevice()
        self.modem = DuplexModem()
        self.modem.device = self.device
        
        self.precoding = PrecodingMock(self.modem)
        self.precoder = PrecoderMock()
        self.precoding[0] = self.precoder
        
    def test_modem_setget(self) -> None:
        """Modem property getter should return setter argument"""
        
        expected_modem = DuplexModem()
        self.precoding.modem = expected_modem
        
        self.assertIs(self.precoding.modem, expected_modem)
        
    def test_required_outputs_validation(self) -> None:
        """Querying the required number of output streams should raise an error if precoding is not set"""
        
        with self.assertRaises(ValueError):
            _ = self.precoding.required_outputs(Mock())
            
    def test_required_outputs(self) -> None:
        """Required number of output streams should report the correct value"""
        
        self.assertEqual(1, self.precoding.required_outputs(self.precoder))
        
        self.modem = ReceivingModem()
        self.modem.device = self.device
        self.precoding.modem = self.modem
        
        self.assertEqual(1, self.precoding.required_outputs(self.precoder))

        self.precoding[1] = PrecoderMock()
        
        self.assertEqual(1, self.precoding.required_outputs(self.precoder))
        
    def test_required_inputs_validation(self) -> None:
        """Querying the required number of input streams should raise an error if precoding is not set"""
        
        with self.assertRaises(ValueError):
            _ = self.precoding.required_outputs(Mock())

    def test_required_outputs(self) -> None:
        """Required number of output streams should report the correct value"""

        self.assertEqual(1, self.precoding.required_outputs(self.precoder))

        self.modem = ReceivingModem()
        self.modem.device = self.device
        self.precoding.modem = self.modem

        self.assertEqual(1, self.precoding.required_outputs(self.precoder))

        self.precoding[1] = PrecoderMock()

        self.assertEqual(1, self.precoding.required_outputs(self.precoder))

    def test_required_inputs_validation(self) -> None:
        """Querying the required number of input streams should raise an error if precoding is not set"""

        with self.assertRaises(ValueError):
            _ = self.precoding.required_inputs(Mock())

    def test_required_inputs(self) -> None:
        """Required number of input streams should report the correct value"""

        self.assertEqual(1, self.precoding.required_inputs(self.precoder))

        self.precoding[1] = PrecoderMock()
        self.assertEqual(1, self.precoding.required_inputs(self.precoding[1]))

    def test_rate(self) -> None:
        """Rate should report the correct value"""

        self.assertEqual(1, self.precoding.rate)

    def test_num_input_streams(self) -> None:
        """Number of input streams should report the correct value"""

        self.assertEqual(1, self.precoding.num_input_streams)

        self.precoding.pop_precoder(0)
        self.assertEqual(1, self.precoding.num_input_streams)

    def test_num_output_streams(self) -> None:
        """Number of output streams should report the correct value"""

        self.assertEqual(1, self.precoding.num_output_streams)

        self.precoding.pop_precoder(0)
        self.assertEqual(1, self.precoding.num_output_streams)

    def test_setitem(self) -> None:
        """Precoders should be properly stored"""

        alpha = PrecoderMock()
        self.precoding[1] = alpha
        self.assertEqual(2, len(self.precoding))

        beta = PrecoderMock()
        self.precoding[0] = beta

        gamma = PrecoderMock()
        self.precoding[-1] = gamma

        self.assertEqual(3, len(self.precoding))
        self.assertIs(gamma, self.precoding[0])
        self.assertIs(beta, self.precoding[1])
        self.assertIs(alpha, self.precoding[2])

    def test_pop_precoder(self) -> None:
        """Precoders should be properly removed"""

        popped_precoder = self.precoding.pop_precoder(0)

        self.assertIs(self.precoder, popped_precoder)
        self.assertEqual(0, len(self.precoding))
