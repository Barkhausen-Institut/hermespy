import unittest

import numpy as np

from modem.rf_chain import RfChain


class TestIqImbalance(unittest.TestCase):

    def test_correct_calculation(self) -> None:
        phase_offset = np.pi
        amplitude_offset = 2

        rf_chain = RfChain(None, phase_offset, amplitude_offset)
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        x_t = i_samples + 1j * q_samples
        expected_detoriated_x_t =  2j*x_t -1j*np.conj(x_t)
        
        np.testing.assert_array_almost_equal(
            expected_detoriated_x_t,
            rf_chain.add_iq_imbalance(x_t)
        )