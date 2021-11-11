from typing import List, Tuple, List
import unittest 

import numpy as np
from scipy.fft import fftshift
from scipy import signal

from hermespy.simulator_core.drop import Drop


class TestDrop(Drop):
    def __init__(self, received_signals: List[np.array],
                 received_bits = [None], received_block_sizes = [0]):
        super().__init__([], [], [], received_signals, received_bits, received_block_sizes)

class TestStoppingCrtiteria(unittest.TestCase):
    def test_receive_stft_properly_calculated_for_non_none_signals(self) -> None:
        received_signals = [np.random.randint(low=0, high=10, size=(1,10))]
        window_size = len(received_signals[0][0])
        f, t, transform = signal.stft(received_signals[0][0], nperseg=window_size,
                            noverlap = int(.5*window_size),
                            return_onesided=False)
        drop = TestDrop(received_signals=received_signals)
        receive_stft = drop.receive_stft
        np.testing.assert_array_almost_equal(
            fftshift(f), receive_stft[0][0]
        )
        np.testing.assert_array_almost_equal(
            t, receive_stft[0][1]
        )
        np.testing.assert_array_almost_equal(
            fftshift(transform, 0), receive_stft[0][2]
        )

    def test_none_stft_for_none_signals(self) -> None:
        received_signals = [
            np.random.randint(low=0, high=10, size=(1,10)),
            None]
        drop = TestDrop(received_signals=received_signals,
                        received_bits=[None, None], received_block_sizes=[0,0])
        receive_stft = drop.receive_stft

        self.assertEquals(receive_stft[1], (None, None, None))