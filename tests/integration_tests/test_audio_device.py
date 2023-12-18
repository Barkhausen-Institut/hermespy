# -*- coding: utf-8 -*-
from unittest import TestCase

from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.fft import fft, fftshift

from hermespy.hardware_loop.audio import AudioDevice
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform, OFDMWaveform, FrameElement, FrameResource, FrameSymbolSection, ElementType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestAudioDevice(TestCase):
    def setUp(self) -> None:
        self.device = AudioDevice(6, 4, [1], [1])

        self.modem = DuplexModem(seed=42)
        self.modem.device = self.device

    @patch("sounddevice.playrec")
    def propagate(self, playrec_mock: MagicMock) -> None:
        def side_effect(*args, **kwargs):
            self.device._AudioDevice__reception = args[0]

        playrec_mock.side_effect = side_effect

        # Execute a fulle device transmit-receive cycle
        _ = self.device.transmit()
        self.device.trigger()
        _ = self.device.receive()

        # Assert the transmit and receive spectra
        transmit_spectrum = fftshift(fft(self.modem.transmission.signal.samples[0, :]))
        receive_spectrum = fftshift(fft(self.modem.reception.signal.samples[0, :]))
        left_bin = int(0.375 * transmit_spectrum.shape[0])
        right_bin = int(0.625 * transmit_spectrum.shape[0])
        assert_array_almost_equal(transmit_spectrum[left_bin:right_bin], receive_spectrum[left_bin:right_bin])
        assert_array_equal(self.modem.transmission.bits, self.modem.reception.bits)

    def test_single_carrier(self) -> None:
        """Test single carrier data transmission over audio devices"""

        waveform = RootRaisedCosineWaveform(symbol_rate=1.2e4, pilot_rate=10, num_preamble_symbols=1, num_data_symbols=100, oversampling_factor=4)
        self.modem.waveform = waveform

        self.propagate()

    def test_ofdm(self) -> None:
        """Test OFDM data transmission over audio devices"""

        resources = [FrameResource(12, 0.01, elements=[FrameElement(ElementType.DATA, 9), FrameElement(ElementType.REFERENCE, 1)])]
        structure = [FrameSymbolSection(3, [0])]

        waveform = OFDMWaveform(subcarrier_spacing=1e2, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure, oversampling_factor=4)
        self.modem.waveform = waveform

        self.propagate()
