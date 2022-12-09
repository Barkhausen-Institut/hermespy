import unittest
import re

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.simulation.rf_chain.rf_chain import RfChain
from hermespy.core.factory import Factory

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
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

        rf_chain = RfChain(None, phase_offset, amplitude_imbalance)

        expected_deteriorated_xt = 0.5j*self.x_t -1j * np.conj(self.x_t)
        
        assert_array_almost_equal(expected_deteriorated_xt, rf_chain.add_iq_imbalance(self.x_t))

    def test_default_values_result_in_no_detoriation(self) -> None:
        rf_chain = RfChain(None, None, None)
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        self.x_t = i_samples + 1j * q_samples
        assert_array_almost_equal(self.x_t, rf_chain.add_iq_imbalance(self.x_t))

    def test_exception_raised_if_amplitude_imbalance_not_within_interval(self) -> None:

        with self.assertRaises(ValueError):
            _ = RfChain(None, None, -3)


class TestIqImbalanceCreationAndSerialization(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = Factory()

    def test_creation_proper_values(self) -> None:

        amplitude_imbalance = 0.5
        phase_offset = 3

        yaml_str = f"""!<RfChain>
       amplitude_imbalance: {amplitude_imbalance}
       phase_offset: {phase_offset}"""

        rf_chain: RfChain = self.factory.from_str(yaml_str)[0]

        self.assertAlmostEqual(rf_chain.amplitude_imbalance, amplitude_imbalance)
        self.assertEqual(rf_chain.phase_offset, phase_offset)

    def test_iq_imbalance_serialisation(self) -> None:

        phase_offset = 10
        amplitude_imbalance = 0.5
        rf_chain = RfChain(phase_offset=phase_offset,
                           amplitude_imbalance=amplitude_imbalance)

        serialized_rf_chain = self.factory.to_str(rf_chain)
        phase_offset_regex = re.compile(f'^phase_offset: {phase_offset}$', re.MULTILINE)
        amplitude_imbalance_regex = re.compile(f'^amplitude_imbalance: {amplitude_imbalance}$', re.MULTILINE)

        self.assertTrue(re.search(phase_offset_regex, serialized_rf_chain) is not None)
        self.assertTrue(re.search(amplitude_imbalance_regex, serialized_rf_chain) is not None)
