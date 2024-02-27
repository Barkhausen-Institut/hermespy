# -*- coding: utf-8 -*-

from __future__ import annotations
from itertools import product
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.modem import DuplexModem, ElementType, GridElement, GridResource, OTFSWaveform, SchmidlCoxPilotSection, SchmidlCoxSynchronization, SymbolSection, TransmittingModem
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestOTFSMWaveform(TestCase):
    """Test the OTFS waveform implementation."""
    
    def setUp(self) -> None:
        
        grid_resources = [
            GridResource(repetitions=10, prefix_ratio=0.1, elements=[GridElement(ElementType.DATA, repetitions=11), GridElement(ElementType.REFERENCE, repetitions=1)]),
            GridResource(repetitions=10, prefix_ratio=0.1, elements=[GridElement(ElementType.DATA, repetitions=5), GridElement(ElementType.REFERENCE, repetitions=1), GridElement(ElementType.DATA, repetitions=6),]),
        ]
        grid_structure = [
            SymbolSection(num_repetitions=2, pattern=[0, 1]),
        ]
        
        self.ofdm = OTFSWaveform(
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            num_subcarriers=128,
            subcarrier_spacing=1e3,
            dc_suppression=False,
            oversampling_factor = 16,
            modulation_order=4,
        )

    def test_transmit_power(self) -> None:
        """Transmitted signal power should approximately be unit"""
        
        modem = TransmittingModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit()
        power = transmission.signal.power
        assert_array_almost_equal(np.ones_like(power) * self.ofdm.power, power, decimal=1)

    def test_transmit_receive(self) -> None:
        """Test the transmission and reception of a signal"""

        modem = DuplexModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit()
        reception = modem.receive(transmission.signal)
        
        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)

    def test_transmit_receive_dc_suppression(self) -> None:
        """Test the DC suppression property"""

        self.ofdm.dc_suppression = True
        modem = DuplexModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit()
        reception = modem.receive(transmission.signal)

        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)
