# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import DuplexModem, ElementType, GridElement, GridResource, OTFSWaveform, SchmidlCoxPilotSection, SchmidlCoxSynchronization, SymbolSection, TransmittingModem
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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

        self.otfs = OTFSWaveform(
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            num_subcarriers=128,
            subcarrier_spacing=1e3,
            dc_suppression=False,
            oversampling_factor=16,
            modulation_order=4,
        )

        self.device = SimulatedDevice()

    def test_transmit_power(self) -> None:
        """Transmitted signal power should approximately be unit"""

        modem = TransmittingModem()
        modem.waveform = self.otfs

        transmission = modem.transmit(self.device.state())
        power = transmission.signal.power
        assert_array_almost_equal(np.ones_like(power) * self.otfs.power, power, decimal=1)

    def test_transmit_receive(self) -> None:
        """Test the transmission and reception of a signal"""

        modem = DuplexModem()
        modem.waveform = self.otfs

        transmission = modem.transmit(self.device.state())
        reception = modem.receive(transmission.signal, self.device.state())

        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)

    def test_transmit_receive_dc_suppression(self) -> None:
        """Test the DC suppression property"""

        self.otfs.dc_suppression = True
        modem = DuplexModem()
        modem.waveform = self.otfs

        transmission = modem.transmit(self.device.state())
        reception = modem.receive(transmission.signal, self.device.state())

        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)

    def test_serialization(self) -> None:
        """Test OTFS waveform serialization"""

        test_roundtrip_serialization(self, self.otfs)
