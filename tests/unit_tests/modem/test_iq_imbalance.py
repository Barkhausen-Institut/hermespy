import unittest
import re

import numpy as np

from modem.rf_chain import RfChain
from simulator_core.factory import Factory


class TestIqImbalance(unittest.TestCase):

    def setUp(self) -> None:
        i_samples = np.random.randint(low=1, high=4, size=100)
        q_samples = np.random.randint(low=1, high=4, size=100)

        self.x_t = i_samples + 1j * q_samples


    def test_correct_calculation(self) -> None:
        phase_offset = np.pi
        amplitude_imbalance = 0.5

        rf_chain = RfChain(None, phase_offset, amplitude_imbalance)

        expected_detoriated_x_t = 0.5j*self.x_t -1j*np.conj(self.x_t)
        
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

    def test_exception_raised_if_amplitude_imbalance_not_within_interval(self) -> None:
        with self.assertRaises(ValueError):
            rf_chain = RfChain(None, None, -3)


class TestIqImbalanceCreationAndSerialization(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = Factory()

    def test_creation_proper_values(self) -> None:
        AMPLITUDE_IMBALANCE = 0.5
        PHASE_OFFSET = 3
        yaml_str = f"""
!<Scenario>

Modems:
  - Transmitter
    RfChain:
       amplitude_imbalance: {AMPLITUDE_IMBALANCE}
       phase_offset: {PHASE_OFFSET}
"""
        scenarios = self.factory.from_str(yaml_str)
        self.assertAlmostEqual(
            scenarios[0].transmitters[0].rf_chain.amplitude_imbalance,
            AMPLITUDE_IMBALANCE
        )
        self.assertEqual(
            scenarios[0].transmitters[0].rf_chain.phase_offset,
            PHASE_OFFSET
        )

    def test_iq_imbalance_serialisation(self) -> None:
        PHASE_OFFSET = 10
        AMPLITUDE_IMBALANCE = 0.5
        rf_chain = RfChain(phase_offset=PHASE_OFFSET,
                           amplitude_imbalance=AMPLITUDE_IMBALANCE)

        serialized_rf_chain = self.factory.to_str(rf_chain)
        phase_offset_regex = re.compile(
            f'^phase_offset: {PHASE_OFFSET}$', re.MULTILINE)
        amplitude_imbalance_regex = re.compile(
            f'^amplitude_imbalance: {AMPLITUDE_IMBALANCE}$',
            re.MULTILINE)

        self.assertTrue(re.search(phase_offset_regex, serialized_rf_chain) is not None)
        self.assertTrue(re.search(amplitude_imbalance_regex, serialized_rf_chain) is not None)
