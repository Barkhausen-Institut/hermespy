import unittest
import re

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.simulation.rf_chain.rf_chain import RfChain
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestIqImbalance(unittest.TestCase):
    def setUp(self) -> None:
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        self.x_t = i_samples + 1j * q_samples

    def test_correct_calculation(self) -> None:
        phase_offset = np.pi
        amplitude_imbalance = 0.5

        rf_chain = RfChain(phase_offset, amplitude_imbalance)

        expected_deteriorated_xt = 0.5j * self.x_t - 1j * np.conj(self.x_t)

        assert_array_almost_equal(expected_deteriorated_xt, rf_chain.add_iq_imbalance(self.x_t))

    def test_default_values_result_in_no_detoriation(self) -> None:
        rf_chain = RfChain(None, None)
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        self.x_t = i_samples + 1j * q_samples
        assert_array_almost_equal(self.x_t, rf_chain.add_iq_imbalance(self.x_t))

    def test_exception_raised_if_amplitude_imbalance_not_within_interval(self) -> None:
        with self.assertRaises(ValueError):
            _ = RfChain(None, -3)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        rf_chain = RfChain()
        test_yaml_roundtrip_serialization(self, rf_chain)
