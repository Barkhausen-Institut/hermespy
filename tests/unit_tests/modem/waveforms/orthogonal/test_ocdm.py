# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import DuplexModem, ElementType, GridElement, GridResource, OCDMWaveform, SymbolSection, TransmittingModem
from hermespy.simulation import SimulatedDevice
from ...test_waveform import TestCommunicationWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestOCDMWaveform(TestCommunicationWaveform):
    """Test the OFDM waveform implementation."""

    waveform: OCDMWaveform

    def setUp(self) -> None:

        grid_resources = [
            GridResource(repetitions=10, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=12)]),
            GridResource(repetitions=12, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=10)]),
        ]
        grid_structure = [
            SymbolSection(num_repetitions=2, pattern=[0, 1]),
        ]

        self.waveform = OCDMWaveform(
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            num_subcarriers=120,
        )

        self.rng = np.random.default_rng(42)
        self.device = SimulatedDevice()

    def test_frame_properties(self) -> None:
        """Ensure the frame signal properties are correct"""

        modem = TransmittingModem()
        modem.waveform = self.waveform
        self.device.transmitters.add(modem)
        waveform = modem.transmit(self.device.state()).signal

        self.assertEqual(self.waveform.samples_per_frame(self.device.bandwidth, self.device.oversampling_factor), waveform.num_samples)

    def test_transmit_receive(self) -> None:
        """Test the transmission and reception of a signal"""

        modem = DuplexModem()
        modem.waveform = self.waveform
        self.device.transmitters.add(modem)

        transmission = modem.transmit(self.device.state())
        reception = modem.receive(transmission.signal, self.device.state())

        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)

    @override
    def test_energy(self) -> None:
        for resource in self.waveform.grid_resources:
            resource.prefix_ratio = 0.0
        super().test_energy()
