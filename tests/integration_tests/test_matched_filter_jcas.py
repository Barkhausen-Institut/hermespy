# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from numpy.random import default_rng


from hermespy.channel import SingleTargetRadarChannel
from hermespy.jcas import MatchedFilterJcas
from hermespy.modem import RootRaisedCosineWaveform, CustomPilotSymbolSequence
from hermespy.modem.waveform_single_carrier import SingleCarrierCorrelationSynchronization, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSCMatchedFilterJcas(TestCase):
    """Test matched filter sensing for psk/qam waveforms."""
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        self.device = SimulatedDevice()
        self.device.carrier_frequency = 1e9
        
        self.target_range = 5
        self.max_range = 10
        self.channel = SingleTargetRadarChannel(target_range=self.target_range,
                                    transmitter=self.device,
                                    receiver=self.device,
                                    radar_cross_section=1.)
        
        self.oversampling_factor = 16

        self.operator = MatchedFilterJcas(self.max_range)
        self.operator.device = self.device
        self.operator.waveform_generator = RootRaisedCosineWaveform(oversampling_factor=self.oversampling_factor, modulation_order=4, num_preamble_symbols=20, num_data_symbols=100, pilot_rate=10, symbol_rate=1e6)
        self.operator.waveform_generator.pilot_symbol_sequence = CustomPilotSymbolSequence(np.array([1, -1, 1j, -1j]))
        self.operator.waveform_generator.synchronization = SingleCarrierCorrelationSynchronization()
        self.operator.waveform_generator.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
        self.operator.waveform_generator.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
        
    def test_jcas(self) -> None:
        """The target distance should be properly estimated while transmitting information."""
        
        for _ in range(5):
            
            # Generate transmitted signal
            transmission = self.device.transmit()
            
            # Propagate signal over the radar channel
            propagetd_signals, _, _ = self.channel.propagate(transmission)
            
            # Receive signal
            self.device.receive(propagetd_signals)
            
            # The bits should be recovered correctly
            assert_array_equal(self.operator.transmission.bits, self.operator.reception.bits)
