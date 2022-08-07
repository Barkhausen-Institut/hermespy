from unittest import TestCase

import numpy as np

from hermespy.channel import Channel, MultipathFading5GTDL
from hermespy.core import Scenario, IdealAntenna, UniformArray
from hermespy.modem.waveform_generator_chirp_fsk import ChirpFSKCorrelationSynchronization, ChirpFSKWaveform
from hermespy.modem.waveform_generator_ofdm import OFDMIdealChannelEstimation, OFDMZeroForcingChannelEqualization
from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, BitErrorEvaluator, RootRaisedCosineWaveform, ZeroForcingChannelEqualization, \
    SingleCarrierCorrelationSynchronization, SingleCarrierZeroForcingChannelEqualization, SingleCarrierIdealChannelEstimation, \
    ChirpFSKWaveform, ChirpFSKCorrelationSynchronization, \
    OFDMWaveform, FrameResource, FrameSymbolSection, FrameElement, ElementType, OFDMCorrelationSynchronization, PilotSection, OFDMLeastSquaresChannelEstimation, OFDMZeroForcingChannelEqualization
from hermespy.precoding import DFT
    
from hermespy.fec import RepetitionEncoder
from hermespy.precoding import SpatialMultiplexing

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSISOLinks(TestCase):
    """Test integration of simulation workflows on the link level for SISO links"""

    def setUp(self) -> None:

        # Configure a 1x1link scenario
        self.tx_device = SimulatedDevice()
        self.rx_device = SimulatedDevice()

        scenario = Scenario(seed=42)
        scenario.add_device(self.tx_device)
        scenario.add_device(self.rx_device)

        # Define a transmit operation on the first device
        self.tx_operator = Modem()
        self.tx_operator.precoding[0] = SpatialMultiplexing()
        self.tx_operator.device = self.tx_device
        self.tx_operator.encoder_manager.add_encoder(RepetitionEncoder())
        self.tx_operator.precoding[1] = DFT()

        # Define a receive operation on the second device
        self.rx_operator = Modem()
        self.rx_operator.precoding[0] = SpatialMultiplexing()
        self.rx_operator.device = self.rx_device
        self.rx_operator.reference_transmitter = self.tx_operator
        self.rx_operator.encoder_manager.add_encoder(RepetitionEncoder())
        self.rx_operator.precoding[1] = DFT()
        
        self.ber = BitErrorEvaluator(self.tx_operator, self.rx_operator)

    def __propagate(self, channel: Channel) -> None:
        """Helper function to propagate a signal from transmitter to receiver.
        
        Args:

            channel (Channel):
                The channel over which to propagate the signal from transmitter to receiver.
        """
        
        channel.set_seed(42)

        transmission = self.tx_operator.transmit()
        tx_signals = self.tx_device.transmit()
        rx_signals, _, channel_state = channel.propagate(tx_signals)
        self.rx_device.receive(np.array([[rx_signals, channel_state]], dtype=object))
        reception = self.rx_operator.receive()
        
        return
    
    def test_ideal_channel_single_carrier(self) -> None:
        """Verify a valid SISO link over an ideal channel with single carrier modulation"""

        tx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
        rx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
        #rx_waveform.synchronization = SingleCarrierCorrelationSynchronization()
        rx_waveform.channel_estimation = SingleCarrierIdealChannelEstimation()
        rx_waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(Channel(self.tx_device, self.rx_device))

        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_tdl_channel_single_carrier(self) -> None:
        """Verify a valid SISO link over a tapped delay line channel with single carrier modulation"""

        tx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, pilot_rate=10, oversampling_factor=8, roll_off=.9)
        rx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, pilot_rate=10, oversampling_factor=8, roll_off=.9)
        #rx_waveform.synchronization = SingleCarrierCorrelationSynchronization()
        rx_waveform.channel_estimation = SingleCarrierIdealChannelEstimation()
        rx_waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(MultipathFading5GTDL(transmitter=self.tx_device, receiver=self.rx_device)) #, doppler_frequency=1e6))

        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_ideal_channel_chirp_fsk(self) -> None:
        """Verify a valid SISO link over an ideal channel with chirp frequency shift keying modulation"""

        tx_waveform = ChirpFSKWaveform()
        rx_waveform = ChirpFSKWaveform()
        rx_waveform.synchronization = ChirpFSKCorrelationSynchronization()

        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        self.tx_operator.precoding.pop_precoder(1)
        self.rx_operator.precoding.pop_precoder(1)
        
        self.__propagate(Channel(self.tx_device, self.rx_device))

        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_tdl_channel_chirp_fsk(self) -> None:
        """Verify a valid SISO link over a tapped delay line channel with chirp frequency shift keying modulation"""
        tx_waveform = ChirpFSKWaveform()
        rx_waveform = ChirpFSKWaveform()
        rx_waveform.synchronization = ChirpFSKCorrelationSynchronization()

        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        self.tx_operator.precoding.pop_precoder(1)
        self.rx_operator.precoding.pop_precoder(1)
        
        self.__propagate(MultipathFading5GTDL(transmitter=self.tx_device, receiver=self.rx_device))
        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_ideal_channel_ofdm(self) -> None:
        """Verify a valid SISO link over an ideal channel ofdm modulation"""
        
        resources = [FrameResource(12, .01, elements=[FrameElement(ElementType.DATA, 9), FrameElement(ElementType.REFERENCE, 1)])]
        structure = [FrameSymbolSection(3, [0])]
        
        tx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        tx_waveform.pilot_section = PilotSection()
        rx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        rx_waveform.pilot_section = PilotSection()
        rx_waveform.synchronization = OFDMCorrelationSynchronization()
        rx_waveform.channel_estimation = OFDMIdealChannelEstimation()
        rx_waveform.channel_equalization = OFDMZeroForcingChannelEqualization()
        
        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(Channel(transmitter=self.tx_device, receiver=self.rx_device))
        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_ideal_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""
        
        resources = [FrameResource(12, .01, elements=[FrameElement(ElementType.DATA, 9), FrameElement(ElementType.REFERENCE, 1)])]
        structure = [FrameSymbolSection(3, [0])]
        
        tx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        tx_waveform.pilot_section = PilotSection()
        rx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        rx_waveform.pilot_section = PilotSection()
        rx_waveform.synchronization = OFDMCorrelationSynchronization()
        rx_waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        rx_waveform.channel_equalization = OFDMZeroForcingChannelEqualization()
        
        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        self.tx_operator.precoding.pop_precoder(1)
        self.rx_operator.precoding.pop_precoder(1)
        
        self.__propagate(Channel(transmitter=self.tx_device, receiver=self.rx_device))
        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_tdl_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over a TDL channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""
        
        resources = [FrameResource(12, .01, elements=[FrameElement(ElementType.DATA, 9), FrameElement(ElementType.REFERENCE, 1)])]
        structure = [FrameSymbolSection(3, [0])]
        
        tx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        tx_waveform.pilot_section = PilotSection()
        rx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        rx_waveform.pilot_section = PilotSection()
        rx_waveform.synchronization = OFDMCorrelationSynchronization()
        rx_waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        rx_waveform.channel_equalization = OFDMZeroForcingChannelEqualization()
        
        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        self.tx_operator.precoding.pop_precoder(1)
        self.rx_operator.precoding.pop_precoder(1)
        
        self.__propagate(MultipathFading5GTDL(transmitter=self.tx_device, receiver=self.rx_device))
        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())


class TestMIMOLinks(TestCase):
    """Test integration of simulation workflow on the link level"""

    def setUp(self) -> None:

        # Configure a 2x2 link scenario
        antennas = UniformArray(IdealAntenna(), 5e-3, [2, 1, 1])
        self.tx_device = SimulatedDevice(antennas=antennas)
        self.rx_device = SimulatedDevice(antennas=antennas)

        scenario = Scenario(seed=42)
        scenario.add_device(self.tx_device)
        scenario.add_device(self.rx_device)

        # Define a transmit operation on the first device
        self.tx_operator = Modem()
        self.tx_operator.precoding[0] = SpatialMultiplexing()
        self.tx_operator.device = self.tx_device
        self.tx_operator.encoder_manager.add_encoder(RepetitionEncoder())

        # Define a receive operation on the second device
        self.rx_operator = Modem()
        self.rx_operator.precoding[0] = SpatialMultiplexing()
        self.rx_operator.device = self.rx_device
        self.rx_operator.reference_transmitter = self.tx_operator
        self.rx_operator.encoder_manager.add_encoder(RepetitionEncoder())

        self.ber = BitErrorEvaluator(self.tx_operator, self.rx_operator)

    def __propagate(self, channel: Channel) -> None:
        """Helper function to propagate a signal from transmitter to receiver.
        
        Args:

            channel (Channel):
                The channel over which to propagate the signal from transmitter to receiver.
        """
        
        channel.set_seed(42)

        transmission = self.tx_operator.transmit()
        tx_signals = self.tx_device.transmit()
        rx_signals, _, channel_state = channel.propagate(tx_signals)
        self.rx_device.receive(np.array([[rx_signals, channel_state]], dtype=object))
        reception = self.rx_operator.receive()
        
        return

    def test_ideal_channel_single_carrier(self) -> None:
        """Verify a valid MIMO link over an ideal channel with single carrier modulation"""

        tx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
        rx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
        rx_waveform.synchronization = SingleCarrierCorrelationSynchronization()
        rx_waveform.channel_estimation = SingleCarrierIdealChannelEstimation()
        rx_waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(Channel(self.tx_device, self.rx_device))

        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_tdl_channel_single_carrier(self) -> None:
        """Verify a valid MIMO link over a tapped delay line channel with single carrier modulation"""

        tx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, pilot_rate=10, oversampling_factor=8, roll_off=.9)
        rx_waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=4, num_data_symbols=40, pilot_rate=10, oversampling_factor=8, roll_off=.9)
        #rx_waveform.synchronization = SingleCarrierCorrelationSynchronization()
        rx_waveform.channel_estimation = SingleCarrierIdealChannelEstimation()
        rx_waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(MultipathFading5GTDL(transmitter=self.tx_device, receiver=self.rx_device)) #, doppler_frequency=1e6))

        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_ideal_channel_ofdm(self) -> None:
        """Verify a valid MIMO link over an ideal channel OFDM modulation"""
        
        resources = [FrameResource(12, .01, elements=[FrameElement(ElementType.DATA, 9), FrameElement(ElementType.REFERENCE, 1)])]
        structure = [FrameSymbolSection(3, [0])]
        
        tx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        tx_waveform.pilot_section = PilotSection()
        rx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        rx_waveform.pilot_section = PilotSection()
        rx_waveform.synchronization = OFDMCorrelationSynchronization()
        rx_waveform.channel_estimation = OFDMIdealChannelEstimation()
        rx_waveform.channel_equalization = OFDMZeroForcingChannelEqualization()
        
        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(Channel(transmitter=self.tx_device, receiver=self.rx_device))
        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
        
    def test_tdl_channel_ofdm(self) -> None:
        """Verify a valid MIMO link over a tapped delay line channel OFDM modulation"""
        
        resources = [FrameResource(12, .01, elements=[FrameElement(ElementType.DATA, 9), FrameElement(ElementType.REFERENCE, 1)])]
        structure = [FrameSymbolSection(3, [0])]
        
        tx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        tx_waveform.pilot_section = PilotSection()
        rx_waveform = OFDMWaveform(subcarrier_spacing=1e3, num_subcarriers=120, dc_suppression=True, resources=resources, structure=structure)
        rx_waveform.pilot_section = PilotSection()
        rx_waveform.synchronization = OFDMCorrelationSynchronization()
        rx_waveform.channel_estimation = OFDMIdealChannelEstimation()
        rx_waveform.channel_equalization = OFDMZeroForcingChannelEqualization()
        
        self.tx_operator.waveform_generator = tx_waveform
        self.rx_operator.waveform_generator = rx_waveform
        
        self.__propagate(MultipathFading5GTDL(transmitter=self.tx_device, receiver=self.rx_device)) #, doppler_frequency=1e6))
        self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())

    # def test_cost256_psk_qam(self) -> None:
    #     """Verify a valid MIMO link over a 3GPP COST256 TDL channel with PSK/QAM modulation"""
    # 
    #     self.tx_operator.waveform_generator = RootRaisedCosineWaveform(oversampling_factor=8)
    #     self.rx_operator.waveform_generator = RootRaisedCosineWaveform(oversampling_factor=8)
    # 
    #     self.__propagate(MultipathFadingCost256(MultipathFadingCost256.TYPE.URBAN, transmitter=self.tx_operator.device, receiver=self.rx_operator.device))
    #     self.assertEqual(0, self.ber.evaluate().artifact().to_scalar())
