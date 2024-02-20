# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import DuplexModem, ElementType, GridElement, GridResource, OCDMWaveform, SymbolSection, TransmittingModem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestOCDMWaveform(TestCase):
    """Test the OFDM waveform implementation."""
    
    def setUp(self) -> None:
        
        grid_resources = [
            GridResource(repetitions=10, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=11), GridElement(ElementType.REFERENCE, repetitions=1)]),
            GridResource(repetitions=10, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=5), GridElement(ElementType.REFERENCE, repetitions=1), GridElement(ElementType.DATA, repetitions=6),]),
        ]
        grid_structure = [
            SymbolSection(num_repetitions=2, pattern=[0, 1]),
        ]
        
        self.ocdm = OCDMWaveform(
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            num_subcarriers=128,
            bandwidth=128e3,
            oversampling_factor = 4,
        )
        
    def test_bandwidth_validation(self) -> None:
        """Bandwidht property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.ocdm.bandwidth = 0.0
            
        with self.assertRaises(ValueError):
            self.ocdm.bandwidth = -1.0
            
    def test_bandwidth_setget(self) -> None:
        """Bandwidth property getter should return setter argument"""
        
        expected_bandwidth = 1.23e4
        self.ocdm.bandwidth = expected_bandwidth
        self.assertEqual(expected_bandwidth, self.ocdm.bandwidth)

    def test_transmit_power(self) -> None:
        """Ensure the transmit power is correct"""
        
        modem = TransmittingModem()
        modem.waveform = self.ocdm
        waveform = modem.transmit().signal
        
        power = waveform.power
        assert_array_almost_equal(np.ones_like(power) * self.ocdm.power, power, decimal=1)

    def test_frame_properties(self) -> None:
        """Ensure the frame signal properties are correct"""
        
        modem = TransmittingModem()
        modem.waveform = self.ocdm
        waveform = modem.transmit().signal
        
        self.assertEqual(self.ocdm.sampling_rate, waveform.sampling_rate)
        self.assertEqual(self.ocdm.bandwidth * self.ocdm.oversampling_factor, waveform.sampling_rate)
        self.assertEqual(self.ocdm.samples_per_frame, waveform.num_samples)

    def test_transmit_receive(self) -> None:
        """Test the transmission and reception of a signal"""

        modem = DuplexModem()
        modem.waveform = self.ocdm
        
        transmission = modem.transmit()
        reception = modem.receive(transmission.signal)
        
        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)
