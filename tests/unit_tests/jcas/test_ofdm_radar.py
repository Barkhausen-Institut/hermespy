# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from scipy.constants import speed_of_light

from hermespy.simulation import SimulatedDevice
from hermespy.jcas import OFDMRadar
from hermespy.modem import ElementType, GridElement, GridResource, OFDMWaveform, SymbolSection

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestOFDMRadar(TestCase):
    """Test the OFDM radar."""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.radar = OFDMRadar(self.device)
        
        resources = [GridResource(64, prefix_ratio=.1, elements=[GridElement(ElementType.DATA, 15), GridElement(ElementType.REFERENCE, 1)])]
        structure = [SymbolSection(11, [0])]
        self.radar.waveform = OFDMWaveform(grid_resources=resources, grid_structure=structure)

    def test_max_range(self) -> None:
        """Max range property should return the correct maximum range"""

        expected_max_range = speed_of_light / (2 * self.radar.waveform.subcarrier_spacing)
        self.assertAlmostEqual(expected_max_range, self.radar.max_range)

    def test_range_resolution(self) -> None:
        """Range resolution property should return the correct range resolution"""

        expected_range_resolution = self.radar.max_range / self.radar.waveform.num_subcarriers
        self.assertAlmostEqual(expected_range_resolution, self.radar.range_resolution)

    def test_transmit_receive(self) -> None:
        """Transmitting and subsequently receiving should return in a correct radar estimate"""

        transmission = self.radar.transmit()
        reception = self.radar.receive(transmission.signal)

        self.assertEqual(0, np.argmax(np.sum(reception.cube.data, (0, 1), keepdims=False)))

    def test_receive_validation(self) -> None:
        """Receive should raise RuntimeError if no cached transmission is available"""

        transmission = self.radar.transmit(cache=False)

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(transmission.signal)
