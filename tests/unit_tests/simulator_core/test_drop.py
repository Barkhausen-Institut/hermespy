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
    def setUp(self) -> None:
        self.received_signals_one_modem = [np.random.randint(low=0, high=10, size=(1,10))]
        self.drop_one_modem = TestDrop(received_signals=self.received_signals_one_modem)
        self.window_size_one_modem = 10

    def test_receive_stft_properly_calculated_for_non_none_signals(self) -> None:
        f, t, transform = signal.stft(
                            self.received_signals_one_modem[0][0],
                            nperseg=self.window_size_one_modem,
                            noverlap = int(.5*self.window_size_one_modem),
                            return_onesided=False)
        receive_stft = self.drop_one_modem.receive_stft
        np.testing.assert_array_almost_equal(
            fftshift(f), receive_stft[0][0]
        )
        np.testing.assert_array_almost_equal(
            t, receive_stft[0][1]
        )
        np.testing.assert_array_almost_equal(
            fftshift(transform, 0), receive_stft[0][2]
        )

    def test_receive_spectrum_properly_calculated_for_non_none_signals(self) -> None:
        f, periodogram = signal.welch(self.received_signals_one_modem[0][0],
                                nperseg=self.window_size_one_modem,
                                noverlap=int(.5*self.window_size_one_modem),
                                return_onesided=False)

        receive_spectrum = self.drop_one_modem.receive_spectrum
        np.testing.assert_array_almost_equal(
            f, receive_spectrum[0][0]
        )
        np.testing.assert_array_almost_equal(
            periodogram, receive_spectrum[0][1]
        )

    def test_receive_spectrum_returns_none_for_none_signals(self) -> None:
        received_signals = [
            np.random.randint(low=0, high=10, size=(1,10)),
            None]
        drop = TestDrop(received_signals=received_signals,
                        received_bits=[None, None],
                        received_block_sizes=[0,0])

        receive_spectrum = drop.receive_spectrum
        self.assertEquals(receive_spectrum[1], (None, None))
        self.assertIsNotNone(receive_spectrum[0][0])
        self.assertIsNotNone(receive_spectrum[0][1])

    def test_none_stft_for_none_signals(self) -> None:
        received_signals = [
            np.random.randint(low=0, high=10, size=(1,10)),
            None]
        drop = TestDrop(received_signals=received_signals,
                        received_bits=[None, None],
                        received_block_sizes=[0,0])
        receive_stft = drop.receive_stft

        self.assertEquals(receive_stft[1], (None, None, None))