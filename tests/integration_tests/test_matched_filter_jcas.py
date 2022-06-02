from unittest import TestCase

from numpy.testing import assert_array_equal
from numpy.random import default_rng


from hermespy.channel import RadarChannel
from hermespy.jcas import MatchedFilterJcas
from hermespy.modem import WaveformGeneratorPskQam, ShapingFilter
from hermespy.modem.waveform_generator_psk_qam import PskQamCorrelationSynchronization, PskQamLeastSquaresChannelEstimation, PskQamZeroForcingChannelEqualization
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPskQamMatchedFilterJcas(TestCase):
    """Test matched filter sensing for psk/qam waveforms."""
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        self.device = SimulatedDevice()
        self.device.carrier_frequency = 1e9
        
        self.target_range = 5
        self.max_range = 10
        self.channel = RadarChannel(target_range=self.target_range,
                                    transmitter=self.device,
                                    receiver=self.device,
                                    radar_cross_section=1.)
        
        self.filter_type = 'ROOT_RAISED_COSINE'
        self.oversampling_factor = 16
        self.modulation_order = 16
        self.guard_interval = 1e-3
        self.filter_length_in_symbols = 16
        self.roll_off_factor = .9

        self.tx_filter = ShapingFilter(filter_type=self.filter_type,
                                       samples_per_symbol=self.oversampling_factor,
                                       is_matched=False,
                                       length_in_symbols=self.filter_length_in_symbols,
                                       roll_off=self.roll_off_factor,
                                       bandwidth_factor=1.)

        self.rx_filter = ShapingFilter(filter_type=self.filter_type,
                                       samples_per_symbol=self.oversampling_factor,
                                       is_matched=True,
                                       length_in_symbols=self.filter_length_in_symbols,
                                       roll_off=self.roll_off_factor,
                                       bandwidth_factor=1.)
        
        self.operator = MatchedFilterJcas(self.max_range)
        self.operator.device = self.device
        self.operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=self.oversampling_factor, num_preamble_symbols=20, num_data_symbols=100,
                                                                   tx_filter=self.tx_filter, rx_filter=self.rx_filter)
        self.operator.waveform_generator.synchronization = PskQamCorrelationSynchronization()
        self.operator.waveform_generator.channel_estimation = PskQamLeastSquaresChannelEstimation()
        self.operator.waveform_generator.channel_equalization = PskQamZeroForcingChannelEqualization()
        
    def test_jcas(self) -> None:
        """The target distance should be properly estimated while transmitting information."""
        
        # Generate transmitted signal
        tx_signal, tx_symbols, tx_bits = self.operator.transmit()
        rf_signals = self.device.transmit()
        
        # Propagate signal over the radar channel
        propagetd_signals, _, _ = self.channel.propagate(rf_signals)
        self.device.receive(propagetd_signals)
        
        # Receive signal
        rx_signal, rx_symbols, rx_bits, radar_cube = self.operator.receive()
        
        # The bits should be recovered correctly
        assert_array_equal(tx_bits, rx_bits)
