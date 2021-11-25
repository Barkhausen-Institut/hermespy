import unittest

import numpy as np
import numpy.testing as nt

from hermespy.precoding.mimo import Mimo


class TestMimo(unittest.TestCase):
    def setUp(self) -> None:
        self.no_streams = 1
        self.no_tx_antennas = 1
        self.no_rx_antennas = 2

        self.K = 8
        self.noise_var = 0.01

        self.channel = self._load_channel()
        self.input = self._load_input()
        self.noise = self._load_noise()

    def test_sc(self) -> None:
        # fixed parameters for testing
        no_rx_antennas = 4
        input_data = self.input[0]

        mimo_SC = Mimo("SC", self.no_streams, self.no_tx_antennas)
        tx_signal = mimo_SC.encode(input_data)

        channel = np.zeros((4, 1, 8), dtype=complex)
        channel[:, 0, :] = self.channel[:, 0, :]

        rx_signal = (
            np.matlib.repmat(tx_signal, no_rx_antennas, 1)
            * np.squeeze(channel, axis=1) + self.noise
        )

        output, channel_estimation, noise_var = mimo_SC.decode(
            rx_signal, channel, self.noise_var)

        output_expected = np.array([[2.486 - 0.9269j, 5.0153-1.8437j, -3.6165-1.879j,
                                        -4.8124-2.3867j, 8.3312+1.7145j, 9.9805+2.0123j,
                                        -8.5141-4.3613j, -9.6395-4.8577j]])

        channel_estimation_expected = np.array([[2.5303 - 0.9431j, 2.5303 - 0.9431j,
                                                    -1.2102 - 0.612j, -1.2102 - 0.6120j,
                                                    1.6620 + 0.3302j, 1.6620 + 0.3302j,
                                                    -1.2102 - 0.6120j, -1.2102 - 0.6120j]])

        noise_var_expected = np.array([[0.0100, 0.0100, 0.0100, 0.0100,
                                        0.0100, 0.0100, 0.0100, 0.0100]])

        nt.assert_array_almost_equal(output, output_expected, decimal=3)
        nt.assert_array_almost_equal(
            channel_estimation, channel_estimation_expected, decimal=3)
        nt.assert_array_almost_equal(noise_var, noise_var_expected, decimal=3)

    def test_mrc(self) -> None:
        # fixed parameters for testing
        no_rx_antennas = 4
        input_data = self.input[0]

        mimo_MRC = Mimo("MRC", self.no_streams, self.no_tx_antennas)
        tx_signal = mimo_MRC.encode(input_data)

        channel = np.zeros((4, 1, 8), dtype=complex)
        channel[:, 0, :] = self.channel[:, 0, :]

        rx_signal = (
            np.matlib.repmat(tx_signal, no_rx_antennas, 1)
            * np.squeeze(channel, axis=1) + self.noise
        )

        output, channel_estimation, noise_var = mimo_MRC.decode(
            rx_signal, channel, self.noise_var)

        output_expected = np.array([[8.2828 - 0.0120j, 16.8081 + 0.0602j, 11.5372 + 0.0703j,
                                        15.5109 + 0.0542j, 18.0656 + 0.0659j,
                                        21.6673 + 0.1634j, 27.5190 + 0.0100j,
                                        31.2993 - 0.0256j]])

        channel_estimation_expected = np.array(
            [[8.4690, 8.4690, 3.9211, 3.9211, 3.5934, 3.5934, 3.9211, 3.9211]])

        noise_var_expected = np.array([[0.0847, 0.0847, 0.0392, 0.0392,
                                        0.0359, 0.0359, 0.0392, 0.0392]])

        nt.assert_array_almost_equal(output, output_expected, decimal=3)
        nt.assert_array_almost_equal(
            channel_estimation, channel_estimation_expected, decimal=3)
        nt.assert_array_almost_equal(noise_var, noise_var_expected, decimal=3)

    def test_stbc_2_antennas(self) -> None:
        # fixed parameters for testing
        no_tx_antennas = 2
        no_rx_antennas = 1

        # symbol_streams = np.zeros((8),dtype=complex)
        input_data = self.input[0, :]

        mimo_stbc = Mimo("STBC", self.no_streams, no_tx_antennas)
        tx_signal = mimo_stbc.encode(input_data)

        channel = np.zeros((1, 2, 8), dtype=complex)
        channel[0, 0, :] = self.channel[0, 0, :]
        channel[0, 1, :] = self.channel[0, 1, :]

        rx_signal = np.zeros((1, self.K), dtype=complex)
        for tx_antenna_idx in range(no_tx_antennas):
            rx_signal += tx_signal[tx_antenna_idx, :] * \
                channel[0, tx_antenna_idx, :]

        output_noiseless, channel_estimation, noise_var = mimo_stbc.decode(
            rx_signal, channel, self.noise_var)

        rx_signal += self.noise[0, :]
        output, channel_estimation, noise_var = mimo_stbc.decode(
            rx_signal, channel, self.noise_var)

        output_expected = np.array([[1.6575 - 0.0212j, 3.2692 - 0.0415j, 1.5872 - 0.0987j,
                                     2.3457 + 0.0055j, 6.9086 + 0.0374j, 8.2958 + 0.0535j,
                                     3.9920 + 0.0537j, 4.7574 + 0.0041j]])
        channel_estimation_expected = np.array(
            [[1.6515, 1.6515, 0.5807, 0.5807, 1.3777, 1.3777, 0.5807, 0.5807]])
        noise_var_expected = np.array(
            [[0.0100,  0.0100,   0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100]])

        nt.assert_array_almost_equal(
            np.real(output_noiseless / channel_estimation),
            np.expand_dims(input_data, 0), decimal=3
        )
        nt.assert_array_almost_equal(output, output_expected, decimal=3)
        nt.assert_array_almost_equal(
            channel_estimation, channel_estimation_expected, decimal=3)
        nt.assert_array_almost_equal(noise_var, noise_var_expected, decimal=3)

    def test_stbc_4_antennas(self) -> None:
        no_tx_antennas = 4
        no_rx_antennas = 1

        input_data = self.input[0]

        mimo_stbc = Mimo("STBC", self.no_streams, no_tx_antennas)
        tx_signal = mimo_stbc.encode(input_data)

        channel = np.zeros((1, 4, 8), dtype=complex)
        channel[0, 0, :] = self.channel[0, 0, :]
        channel[0, 1, :] = self.channel[0, 1, :]
        channel[0, 2, :] = self.channel[0, 2, :]
        channel[0, 3, :] = self.channel[0, 3, :]

        rx_signal = np.zeros((1, self.K), dtype=complex)
        for tx_antenna_idx in range(no_tx_antennas):
            rx_signal += tx_signal[tx_antenna_idx, :] * \
                channel[0, tx_antenna_idx, :]

        output_noiseless, channel_estimation, noise_var = mimo_stbc.decode(
            rx_signal, channel, self.noise_var)

        rx_signal += self.noise[0, :]
        output, channel_estimation, noise_var = mimo_stbc.decode(
            rx_signal, channel, self.noise_var)

        output_expected = np.array([[0.6632 - 0.0044j, 1.4229 - 0.0262j, 2.0186 + 0.0179j,
                                     2.6283 + 0.0387j, 6.2880 + 0.0514j, 7.4944 + 0.0357j,
                                     4.7837 - 0.0012j, 5.5557 - 0.0969j]])

        channel_estimation_expected = np.array(
            [[0.7143, 0.7143, 0.6980, 0.6980, 1.2496, 1.2496, 0.6980, 0.6980]])
        noise_var_expected = np.array(
            [[0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100]])

        nt.assert_array_almost_equal(
            channel_estimation, channel_estimation_expected, decimal=3)
        nt.assert_array_almost_equal(
            output_noiseless/channel_estimation, np.expand_dims(input_data, 0), decimal=3)
        nt.assert_array_almost_equal(output, output_expected, decimal=3)
        nt.assert_array_almost_equal(noise_var, noise_var_expected, decimal=3)

    def test_sm_mmse(self) -> None:
        no_streams = 4
        no_tx_antennas = 4
        no_rx_antennas = 4

        mimo_sm_mmse = Mimo("SM-MMSE", no_streams, no_tx_antennas)

        tx_signal_sm = mimo_sm_mmse.encode(self.input)

        rx_signal = np.zeros((no_rx_antennas, self.K), dtype=complex)

        for rx_antenna_idx in range(no_rx_antennas):
            for tx_antenna_idx in range(no_tx_antennas):
                rx_signal[rx_antenna_idx, :] += (
                    tx_signal_sm[tx_antenna_idx]
                    * self.channel[rx_antenna_idx, tx_antenna_idx, :].ravel()
                )

        rx_signal += self.noise
        output, channel_estimation, noise_var = mimo_sm_mmse.decode(
            rx_signal, self.channel, self.noise_var)

        output_expected = np.array([[0.98926-0.02114j, 2.00785+0.00385j, -5.05308-4.91580j, -4.62477-5.13813j,
                                     6.10257-0.51518j, 7.15567-0.49804j, -2.87335-5.98062j, -2.07398-6.14289j],
                                    [9.03150+0.00426j, 10.08268+0.08995j, 9.84237+1.15782j, 10.74988+1.40029j,
                                     11.82801+0.25859j, 12.79893+0.23293j, 13.60707+1.61753j, 14.57386+1.72007j],
                                    [17.50727+0.07995j, 18.44156+0.18392j, 8.37648-16.82162j, 7.38566-16.95741j,
                                     19.75725-0.58267j, 20.63052-0.71848j, 6.98928-19.69763j, 7.97153-19.93819j],
                                    [25.37531-0.05401j, 26.43067-0.18321j, 23.50777+5.92758j, 24.14454+6.60229j,
                                     28.14077+0.43227j, 29.14565+0.45121j, 26.14762+7.91874j, 27.02473+ 8.19929j]])

        channel_estimation_expected = np.ones((no_streams, self.K))

        noise_var_expected = np.array([[0.0015774, 0.0015774, 0.1668822, 0.1668822,
                                        0.0151182, 0.0151182, 0.1668822, 0.1668822],
                                       [0.0045647, 0.0045647, 0.0106691, 0.0106691,
                                        0.0218888, 0.0218888, 0.0106691, 0.0106691],
                                       [0.0363676, 0.0363676, 1.3020959, 1.3020959,
                                        0.0668707, 0.0668707, 1.3020959, 1.3020959],
                                       [0.0164030, 0.0164030, 0.1757167, 0.1757167,
                                        0.0285865, 0.0285865, 0.1757167, 0.1757167]])
        nt.assert_array_almost_equal(
            channel_estimation, channel_estimation_expected, decimal=3)
        nt.assert_array_almost_equal(output, output_expected, decimal=3)
        #nt.assert_array_almost_equal(noise_var, noise_var_expected, decimal=3)

    def test_sm_zf(self) -> None:
        no_streams = 4
        no_tx_antennas = 4
        no_rx_antennas = 4

        mimo_sm_zf = Mimo("SM-ZF", no_streams, no_tx_antennas)

        tx_signal_sm = mimo_sm_zf.encode(self.input)

        rx_signal = np.zeros((no_rx_antennas, self.K), dtype=complex)

        for rx_antenna_idx in range(no_rx_antennas):
            for tx_antenna_idx in range(no_tx_antennas):
                rx_signal[rx_antenna_idx, :] += (
                    tx_signal_sm[tx_antenna_idx]
                    * self.channel[rx_antenna_idx, tx_antenna_idx, :].ravel()
                )

        output_noiseless, channel_estimation, noise_var = mimo_sm_zf.decode(
            rx_signal, self.channel, self.noise_var)

        rx_signal += self.noise
        output, channel_estimation, noise_var = mimo_sm_zf.decode(
            rx_signal, self.channel, self.noise_var)

        output_expected = np.array([[0.97351-0.01263j, 1.99243+0.01226j, 3.29398+0.13902j, 2.90919+0.46548j,
                                     4.97826+0.02467j, 5.98248+0.07030j, 5.48199-0.73530j, 7.74690-0.19440j],
                                    [8.9827-0.0513j, 10.0354+0.0323j, 11.0221-0.1556j, 11.9842+0.2661j,
                                     13.0474-0.0074j, 14.0536-0.0543j, 14.7767+0.2446j, 15.9572+0.0677j],
                                    [17.063-0.093j, 17.982+0.015j, 19.754-0.017j, 17.903+1.101j,
                                     21.144+0.073j, 22.069-0.049j, 19.786-1.239j, 23.504-0.262j],
                                    [24.972 + 0.082j, 26.000 - 0.047j, 27.261 - 0.299j, 27.980 + 1.194j,
                                     29.087 + 0.001j, 30.127 - 0.009j, 29.683 + 1.117j, 31.719 + 0.157j]])

        channel_estimation_expected = np.ones((no_streams, self.K))

        noise_var_expected = np.array([[0.0015865, 0.0015865, 0.9465638, 0.9465638,
                                        0.0165235, 0.0165235, 0.9465638, 0.9465638],
                                       [0.0047451, 0.0047451, 0.0453697, 0.0453697,
                                        0.0239917, 0.0239917, 0.0453697, 0.0453697],
                                       [0.0370233, 0.0370233, 3.8285351, 3.8285351,
                                        0.0708514, 0.0708514, 3.8285351, 3.8285351],
                                       [0.0169841, 0.0169841, 0.9910722, 0.9910722,
                                        0.0312523, 0.0312523, 0.9910722, 0.9910722]])

        nt.assert_array_almost_equal(
            output_noiseless / channel_estimation, self.input, decimal=3)
        nt.assert_array_almost_equal(output, output_expected, decimal=3)
        nt.assert_array_almost_equal(
            channel_estimation, channel_estimation_expected, decimal=3)
        nt.assert_array_almost_equal(noise_var, noise_var_expected, decimal=3)

    def _load_noise(self) -> np.ndarray:
        # by default the input has no_rx_antennas = 4, N = 8
        noise = np.array([[0.0433 - 1j*0.0140, 0.0176 + 1j*0.0311, 0.1279 + 0.0065j,
                                -0.0512 - 1j*0.1237, 0.0212 + 1j*0.0635, 0.0085 + 1j*0.0311,
                                0.0537 - 1j*0.1311, -0.0217 - 1j*0.0119],
                          [-0.0039 + 1j*0.0232, -0.0702 - 1j*0.0436, -0.0764 + 0.1223j,
                                -0.0419 + 1j*0.0644, -0.0264 + 1j*0.0130, 0.0404 + 1j*0.0072,
                                -0.0465 - 1j*0.0807, -0.0093 - 1j*0.0155],
                          [-0.0791 - 1j*0.0169, 0.0689 + 1j*0.0194, 0.0141 - 0.0430j,
                                0.0284 + 1j*0.0613, 0.0577 + 1j*0.0206, 0.0292 + 1j*0.1971,
                                -0.0427 - 1j*0.0773, 0.0421 + 1j*0.0383],
                          [-0.0443 + 1j*0.0162, -0.0453 + 1j*0.0425, -0.1076 - 0.0521j,
                                0.0666 - 1j*0.0056, 0.0565 + 1j*0.0080, -0.0698 - 1j*0.0825,
                                0.0125 - 1j*0.0307, 0.0740 + 1j*0.0275]])

        return noise

    def _load_input(self) -> np.ndarray:
        # by default the input has no_tx_antennas = 4, N = 8
        input_signal = np.array([np.arange(1, 9), np.arange(9, 17),
                          np.arange(17, 25), np.arange(25, 33)])
        return input_signal

    def _load_channel(self) -> np.ndarray:
        # by defaul the channel has no_rx_antennas = 4, no_tx_antennas = 4, N = 8
        channel = np.zeros((4, 4, 8), dtype=complex)

        channel[:, :, 0] = np.array([[-0.9247 + 1j*0.3592, 1.9583 + 1j*0.7973,
                                        -0.0446 - 1j*0.18531, 1.0534 - 1j*0.6924],
                                     [-0.3066 + 1j*0.1994, -0.9545 + 1j*0.2476,
                                        0.5054 - 1j*1.2376, 0.9963 - 1j*0.8177],
                                     [0.2423 + 1j*0.0237, 2.1460 - 1j*0.2115,
                                        -0.1449 - 1j*0.2020, 1.0021 - 1j*0.3773],
                                     [2.5303 - 1j*0.9431, 0.5129 + 1j*0.0162,
                                        -0.0878 - 1j*0.5879, 0.4748 - 1j*1.4161]])

        channel[:, :, 1] = channel[:, :, 0]

        channel[:, :, 2] = np.array([[-0.5338 + 0.2175j, -0.1707 + 1j*0.5596,
                                        -0.0212 + 1j*0.2358, 0.7844 + 1j*0.1299],
                                     [0.9689 - 0.8889j, 0.2257 - 1j*0.9419,
                                        -0.1166 + 1j*0.2767, -0.6107 - 1j*0.3367],
                                     [-1.2102 - 0.6120j, 0.2212 - 1j*1.6475,
                                        0.4439 + 1j*0.3194, 0.0547 + 1j*0.6095],
                                     [-0.0723 - 0.1248j, -0.6116 - 1j*1.0247,
                                        0.7731 - 1j*0.0921, -0.8585 - 1j*0.9629]])

        channel[:, :, 3] = channel[:, :, 2]

        channel[:, :, 4] = np.array([[1.6620 + 1j*0.3302, 0.6283 - 1j*0.7282, 
                                        0.3452 + 1j*0.3643, 0.2062 - 1j*0.1033],
                                     [-0.4353 - 1j*0.1483, -0.5408 + 1j*0.6712,
                                        -0.1254 + 1j*0.1848, 0.1399 - 1j*0.3762],
                                     [0.5290 + 1j*0.4421, -0.9916 + 1j*0.2171,
                                        -0.1386 - 1j*0.6657, 1.1227 + 1j*1.1894],
                                     [-0.1361 + 1j*0.1296, -1.0058 + 1j*0.0956,
                                         1.0036 - 1j*0.1148, -0.5688 - 1j*0.6192]])

        channel[:, :, 5] = channel[:, :, 4]

        channel[:, :, 6] = np.array([[-0.5338 + 1j*0.2175, -0.1707 + 1j*0.5596,
                                        -0.0212 + 1j*0.2358, 0.7844 + 1j*0.1299],
                                     [0.9689 - 1j*0.8889, 0.2257 - 1j*0.9419,
                                        -0.1166 + 1j*0.2767, -0.6107 - 1j*0.3367],
                                     [-1.2102 - 1j*0.6120, 0.2212 - 1j*1.6475,
                                        0.4439 + 1j*0.3194, 0.0547 + 1j*0.6095],
                                     [-0.0723 - 1j*0.1248, -0.6116 - 1j*1.0247,
                                        0.7731 - 1j*0.0921, -0.8585 - 1j*0.9629]])

        channel[:, :, 7] = channel[:, :, 6]

        channel = np.asarray(channel)
        return channel


if __name__ == '__main__':
    unittest.main()
