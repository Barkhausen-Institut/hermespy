# -*- coding: utf-8 -*-

from __future__ import annotations
from itertools import product
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.modem import DuplexModem, ElementType, GridElement, GridResource, OFDMWaveform, SchmidlCoxPilotSection, SchmidlCoxSynchronization, SymbolSection, TransmittingModem
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestOFDMWaveform(TestCase):
    """Test the OFDM waveform implementation."""
    
    def setUp(self) -> None:
        
        grid_resources = [
            GridResource(repetitions=10, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=11), GridElement(ElementType.REFERENCE, repetitions=1)]),
            GridResource(repetitions=10, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=5), GridElement(ElementType.REFERENCE, repetitions=1), GridElement(ElementType.DATA, repetitions=6),]),
        ]
        grid_structure = [
            SymbolSection(num_repetitions=2, pattern=[0, 1], sample_offset=3),
        ]
        
        self.ofdm = OFDMWaveform(
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            num_subcarriers=128,
            subcarrier_spacing=1e3,
            dc_suppression=False,
            oversampling_factor = 4,
        )
        
        self.device = SimulatedDevice()
        
    def test_subcarrier_spacing_validation(self) -> None:
        """Subcarrier spacing property should raise a value error for invalid values"""
        
        with self.assertRaises(ValueError):
            self.ofdm.subcarrier_spacing = -1.0
            
        with self.assertRaises(ValueError):
            self.ofdm.subcarrier_spacing = 0.0
    
    def test_subcarrier_spacing_setget(self) -> None:
        """Subcarrier spacing property getter should return setter argument"""
        
        expected_spacing = 1.23e4
        self.ofdm.subcarrier_spacing = expected_spacing
        self.assertEqual(expected_spacing, self.ofdm.subcarrier_spacing)
        
    def test_samples_per_frame(self) -> None:
        """Samples per frame should be the expected value"""
        
        pilot_section = Mock()
        pilot_section.num_samples = 123
        self.ofdm.pilot_section = pilot_section
 
        expected_num_samples = 123 + (512 + int(512 * .01)) * 4
        self.assertEqual(expected_num_samples, self.ofdm.samples_per_frame)

    def test_frame_properties(self) -> None:
        """Ensure the frame signal properties are correct"""
        
        modem = TransmittingModem()
        modem.waveform = self.ofdm
        waveform = modem.transmit(self.device.state()).signal
        
        self.assertEqual(self.ofdm.sampling_rate, waveform.sampling_rate)
        self.assertEqual(self.ofdm.bandwidth * self.ofdm.oversampling_factor, waveform.sampling_rate)
        self.assertEqual(self.ofdm.samples_per_frame, waveform.num_samples)

    def test_transmit_power(self) -> None:
        """Transmitted signal power should approximately be unit"""
        
        modem = TransmittingModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit(self.device.state())
        power = transmission.signal.power
        assert_array_almost_equal(np.ones_like(power) * self.ofdm.power, power, decimal=1)

    def test_transmit_receive_dc_suppression(self) -> None:
        """Test the DC suppression property"""

        self.ofdm.dc_suppression = True
        modem = DuplexModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit(self.device.state())
        reception = modem.receive(transmission.signal, self.device.state())
        
        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)

        # Make sure DC is actually suppressed
        dc = np.mean(transmission.signal[0, :])
        self.assertAlmostEqual(0, dc, delta=1e-2)
        
    def test_transmit_receive(self) -> None:
        """Test the transmission and reception of a signal"""

        modem = DuplexModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit(self.device.state())
        reception = modem.receive(transmission.signal ,self.device.state())
        
        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)

    def test_transmit_receive_dc_suppression(self) -> None:
        """Test the DC suppression property"""

        self.ofdm.dc_suppression = True
        modem = DuplexModem()
        modem.waveform = self.ofdm
        
        transmission = modem.transmit(self.device.state())
        reception = modem.receive(transmission.signal, self.device.state())

        # Make sure the symbols are properly received
        assert_array_almost_equal(transmission.symbols.raw, reception.equalized_symbols.raw)
        
    def test_serialization(self) -> None:
        """Test OFDM waveform serialization"""

        test_roundtrip_serialization(self, self.ofdm, {'modem'})


class TestSchmidlCoxPilotSection(TestCase):
    """Test the Schmidl Cox Algorithm Pilot section implementation."""

    def setUp(self) -> None:
        self.pilot = SchmidlCoxPilotSection()

    def test_serialization(self) -> None:
        """Test schmidl cox pilot section serialization"""

        test_roundtrip_serialization(self, self.pilot)


class TestSchmidlCoxSynchronization(TestCase):
    """Test the Schmidl Cox Algorithm implementation."""

    def setUp(self) -> None:
        self.rng = default_rng(90)

        test_resource = GridResource(repetitions=10, prefix_ratio=0.01, elements=[GridElement(ElementType.DATA, repetitions=11), GridElement(ElementType.REFERENCE, repetitions=1)])
        test_payload = SymbolSection(num_repetitions=6, pattern=[0])
        self.frame = OFDMWaveform(oversampling_factor=1, grid_resources=[test_resource], grid_structure=[test_payload], num_subcarriers=120)
        self.frame.pilot_section = SchmidlCoxPilotSection()
        self.frame.dc_suppression = False

        self.synchronization = SchmidlCoxSynchronization(self.frame)

        self.num_streams = 3
        self.delays_in_samples = [0, 9, 80]
        self.num_frames = [1, 2, 3]
        self.max_delay_offset = 8

    def test_synchronize_empty_signal(self) -> None:
        """Test the proper estimation of delays during correlation synchronization for an empty signal"""

        signal = np.empty((self.num_streams, 0), dtype=np.complex128)
        frame_delays = self.synchronization.synchronize(signal)

        self.assertEqual(0, len(frame_delays))

    def test_synchronize(self) -> None:
        """Test the proper estimation of delays during Schmidl-Cox synchronization"""

        for d, n in product(self.delays_in_samples, self.num_frames):
            symbols = [self.frame.map(self.rng.integers(0, 2, size=self.frame.bits_per_frame(self.frame.num_data_symbols))) for _ in range(n)]
            frames = [np.outer(np.exp(2j * pi * self.rng.uniform(0, 1, (self.num_streams, 1))), self.frame.modulate(symbols[f])) for f in range(n)]

            signal = np.empty((self.num_streams, 0), dtype=np.complex128)
            for frame in frames:
                signal = np.concatenate((signal, np.zeros((self.num_streams, d), dtype=np.complex128), frame), axis=1)

            estimated_delays = self.synchronization.synchronize(signal)
            self.assertEqual(n, len(estimated_delays))

            for f, estimated_delay in enumerate(estimated_delays):
                expected_delay = f * self.frame.samples_per_frame + (1 + f) * d

                if abs(estimated_delay - expected_delay) > self.max_delay_offset:
                    self.fail(f"Estimated delay {estimated_delay} too far off (should be {expected_delay})")

    def test_serialization(self) -> None:
        """Test schmiodl cox synchronization serialization"""

        test_roundtrip_serialization(self, self.synchronization, {'waveform'})
