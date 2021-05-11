import unittest
import os
import numpy as np

from simulator_core.random_streams import RandomStreams


class TestRandomStreams(unittest.TestCase):
    def setUp(self) -> None:
        seed = 10
        self.rnd = RandomStreams(seed)

    def test_if_matlab_file_exists(self) -> None:
        file_path = os.path.join(
            os.getcwd(), RandomStreams.seeds_dir_name, RandomStreams.seeds_file_name
        )
        self.assertTrue(os.path.exists(file_path))

    def test_seed_generation(self) -> None:

        # test if different instances are generated from different seeds
        state_noise = self.rnd._rng_dict['noise'].get_state()
        state_channel = self.rnd._rng_dict['channel'].get_state()
        state_source = self.rnd._rng_dict['source'].get_state()
        state_hardware = self.rnd._rng_dict['hardware'].get_state()

        self.assertFalse(np.array_equal(state_noise[1], state_channel[1]))
        self.assertFalse(np.array_equal(state_noise[1], state_source[1]))
        self.assertFalse(np.array_equal(state_noise[1], state_hardware[1]))
        self.assertFalse(np.array_equal(state_channel[1], state_source[1]))
        self.assertFalse(np.array_equal(state_channel[1], state_hardware[1]))
        self.assertFalse(np.array_equal(state_source[1], state_hardware[1]))

    def test_invalid_seed_type(self) -> None:
        invalid_seed_type = "InvalidSeedType"

        self.assertRaises(
            ValueError,
            lambda: self.rnd.get_rng(invalid_seed_type))


if __name__ == '__main__':

    os.chdir("../..")
    unittest.main()
