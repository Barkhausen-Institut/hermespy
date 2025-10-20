# -*- coding: utf-8 -*-
from unittest import TestCase

from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.fft import fft, fftshift

from hermespy.hardware_loop.audio import AudioDevice
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform, OFDMWaveform, GridElement, GridResource, SymbolSection, ElementType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestAudioDevice(TestCase):
    def setUp(self) -> None:
        self.device = AudioDevice(6, 4, [1], [1])

        self.modem = DuplexModem(seed=42)
        self.device.add_dsp(self.modem)

    @patch("sounddevice.playrec")
    def propagate(self, playrec_mock: MagicMock) -> None:
        def side_effect(*args, **kwargs):
            self.device._AudioDevice__reception = args[0]

        playrec_mock.side_effect = side_effect

        # Execute a fulle device transmit-receive cycle
        transmission = self.device.transmit().operator_transmissions[0]
        self.device.trigger()
        reception = self.device.receive().operator_receptions[0]

        # Assert the transmit and receive spectra
        transmit_spectrum = fftshift(fft(transmission.signal))
        receive_spectrum = fftshift(fft(reception.signal))
        left_bin = int(0.375 * transmit_spectrum.shape[0])
        right_bin = int(0.625 * transmit_spectrum.shape[0])
        assert_array_almost_equal(transmit_spectrum[left_bin:right_bin], receive_spectrum[left_bin:right_bin])
        assert_array_equal(transmission.bits, reception.bits)

    def test_single_carrier(self) -> None:
        """Test single carrier data transmission over audio devices"""

        waveform = RootRaisedCosineWaveform(pilot_rate=10, num_preamble_symbols=1, num_data_symbols=100)
        self.modem.waveform = waveform

        self.propagate()

    def test_ofdm(self) -> None:
        """Test OFDM data transmission over audio devices"""

        resources = [GridResource(12, prefix_ratio=0.0, elements=[GridElement(ElementType.DATA, 9), GridElement(ElementType.REFERENCE, 1)])]
        structure = [SymbolSection(3, [0])]

        waveform = OFDMWaveform(
            num_subcarriers=128,
            dc_suppression=True,
            grid_resources=resources,
            grid_structure=structure,
            modulation_order=4,
        )
        self.modem.waveform = waveform

        self.propagate()
