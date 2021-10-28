import unittest

import numpy as np

from modem.rf_chain import RfChain


class TestIqImbalance(unittest.TestCase):

    def setUp(self) -> None:
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        self.x_t = i_samples + 1j * q_samples
    def test_correct_calculation(self) -> None:
        phase_offset = np.pi
        amplitude_offset = 2

        rf_chain = RfChain(None, phase_offset, amplitude_offset)

        expected_detoriated_x_t =  2j*self.x_t -1j*np.conj(self.x_t)
        
        np.testing.assert_array_almost_equal(
            expected_detoriated_x_t,
            rf_chain.add_iq_imbalance(self.x_t)
        )

    def test_default_values_result_in_no_detoriation(self) -> None:
        rf_chain = RfChain(None, None, None)
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        self.x_t = i_samples + 1j * q_samples
        np.testing.assert_array_almost_equal(
            self.x_t, rf_chain.add_iq_imbalance(self.x_t)
        )