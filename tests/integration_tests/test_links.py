# -*- coding: utf-8 -*-

from copy import deepcopy
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.channel import Channel, IdealChannel, TDL, Cost259, RandomDelayChannel, TDLType, UrbanMicrocells
from hermespy.core import Transformation
from hermespy.simulation import SimulatedDevice, SimulationScenario, SingleCarrierIdealChannelEstimation, OFDMIdealChannelEstimation, SimulatedUniformArray, SimulatedIdealAntenna
from hermespy.modem import (
    SimplexLink,
    BitErrorEvaluator,
    RootRaisedCosineWaveform,
    SingleCarrierCorrelationSynchronization,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
    ChirpFSKWaveform,
    ChirpFSKCorrelationSynchronization,
    OCDMWaveform,
    OFDMWaveform,
    OTFSWaveform,
    GridResource,
    SymbolSection,
    GridElement,
    ElementType,
    OFDMCorrelationSynchronization,
    PilotSection,
    OrthogonalLeastSquaresChannelEstimation,
    OrthogonalZeroForcingChannelEqualization,
    SchmidlCoxPilotSection,
    SchmidlCoxSynchronization,
    ReferencePosition,
    OTFSWaveform,
    Synchronization,
)
from hermespy.fec import RepetitionEncoder, BlockInterleaver

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestLinksBase(TestCase):
    """Base class for link integration tests."""

    tx_device: SimulatedDevice
    rx_device: SimulatedDevice
    ber: BitErrorEvaluator
    link: SimplexLink
    _doppler_frequency: float = 10

    def setUp(self) -> None:
        # Configure a 1x1link scenario
        scenario = SimulationScenario(seed=42)
        carrier_frequency = 2.4e9
        self.tx_device = scenario.new_device(
            pose=Transformation.From_Translation(np.array([0, 0, 50])),
            carrier_frequency=carrier_frequency,
        )
        self.rx_device = scenario.new_device(
            pose=Transformation.From_Translation(np.array([80, 80, 20])),
            carrier_frequency=carrier_frequency,
        )
        self.rx_device.velocity = 1 * (self.rx_device.position - self.tx_device.position) / np.linalg.norm(self.rx_device.position - self.tx_device.position)

        # Define a simplex linke between the two devices
        self.link = SimplexLink()
        self.tx_device.transmitters.add(self.link)
        self.rx_device.receivers.add(self.link)
        # self.link.precoding[0] = DFT()

        self.repeater = RepetitionEncoder(bit_block_size=16)
        self.link.encoder_manager.add_encoder(self.repeater)

        self.interleaver = BlockInterleaver(block_size=10 * 16, interleave_blocks=4)
        self.link.encoder_manager.add_encoder(self.interleaver)

        # Specify a bit error evaluator
        self.ber = BitErrorEvaluator(self.link, self.link)

    def __propagate(self, channel: Channel) -> None:
        """Helper function to propagate a signal from transmitter to receiver.

        Args:

            channel (Channel):
                The channel over which to propagate the signal from transmitter to receiver.
        """

        channel.seed = 42
        self.link.seed = 42

        self.device_transmission = self.tx_device.transmit()
        channel_realization = channel.realize()
        self.channel_sample = channel_realization.sample(self.tx_device, self.rx_device)
        channel_propagation = self.channel_sample.propagate(self.device_transmission)
        self.rx_device.receive(channel_propagation)

        # Debug:
        #
        # link_transmission = device_transmission.operator_transmissions[0]
        # tx = link_transmission.signal
        # state = self.channel_sample.state(tx.num_samples, tx.num_samples)
        # rx_prediction = state.propagate(tx)

    # =======================
    # Waveform configurations
    # =======================

    def __configure_single_carrier_waveform(self, channel: Channel) -> RootRaisedCosineWaveform:
        """Configure a single carrier wafeform with default parameters.

        Returns: The configured waveform.
        """

        waveform = RootRaisedCosineWaveform(symbol_rate=1 / 10e-6, num_preamble_symbols=10, num_data_symbols=160, pilot_rate=10, oversampling_factor=8, roll_off=0.9)
        waveform.synchronization = SingleCarrierCorrelationSynchronization()
        waveform.channel_estimation = SingleCarrierIdealChannelEstimation(channel, self.tx_device, self.rx_device)
        waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
        self.link.waveform = waveform

        return waveform

    def __configure_chirp_fsk_waveform(self) -> ChirpFSKWaveform:
        """Configure a chirp frequency shift keying wafeform with default parameters.

        Returns: The configured waveform.
        """

        waveform = ChirpFSKWaveform(chirp_duration=1e-5, chirp_bandwidth=375e6)
        waveform.synchronization = ChirpFSKCorrelationSynchronization()
        self.link.waveform = waveform

        return waveform

    def __configure_ocdm_waveform(self, channel: Channel) -> OCDMWaveform:
        """Configure an OCDM waveform with default parameters.

        Returns: The configured waveform.
        """

        grid_resources = [
            GridResource(50, prefix_ratio=.4, elements=[GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.DATA, 1)]),
            GridResource(50, prefix_ratio=.4, elements=[GridElement(ElementType.DATA, 1), GridElement(ElementType.REFERENCE, 1)]),
        ]
        grid_structure = [
            SymbolSection(
                5,
                [0, 1],
                0,
            ),
        ]

        waveform = OCDMWaveform(5e4, 100, grid_resources, grid_structure, modulation_order=4)
        waveform.oversampling_factor = 2
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()


        self.link.waveform = waveform

        # Debugging: Deactivate error correction
        self.repeater.enabled = False
        self.interleaver.enabled = False

        return waveform

    def __configure_ofdm_waveform(self, channel: Channel) -> OFDMWaveform:
        """Configure an OFDM wafeform with default parameters.

        Returns: The configured waveform.
        """

        # Mock 5G numerology #1:
        # 120khz subcarrier spacing, 120 subcarriers, 2us guard interval, 1ms subframe duration

        prefix_ratio = 0.0684
        num_subcarriers = 128
        grid_resources = [
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 4)
            ]),
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.DATA, 2),
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 2),
            ]),
        ]
        grid_structure = [
            SymbolSection(
                num_subcarriers // 2,
                [0, 1],
                5,
            ),
        ]

        waveform = OFDMWaveform(subcarrier_spacing=1e6 / num_subcarriers, num_subcarriers=num_subcarriers, dc_suppression=True, grid_resources=grid_resources, grid_structure=grid_structure)
        waveform.oversampling_factor = 2
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OFDMIdealChannelEstimation(channel, self.tx_device, self.rx_device, reference_position=ReferencePosition.IDEAL)
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()
        #waveform.pilot_symbol_sequence = CustomPilotSymbolSequence(np.arange(1, num_subcarriers * 60))
        self.link.waveform = waveform

        # Properly configure the error correction
        #bits_per_symbol = waveform.bits_per_frame() // 15
        #self.repeater.bit_block_size = bits_per_symbol // self.repeater.repetitions
        #self.interleaver.block_size = bits_per_symbol
        #self.interleaver.interleave_blocks = waveform.bits_per_symbol

        # Debugging: Deactivate error correction
        self.repeater.enabled = False
        self.interleaver.enabled = False

        return waveform

    def __configure_otfs_waveform(self, channel: Channel) -> OTFSWaveform:
        """Configure an OTFS waveform with default parameters.

        Returns: The configured waveform.
        """

        prefix_ratio = 2 * 0.0684
        num_subcarriers = 128
        grid_resources = [
            GridResource(num_subcarriers // 4, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.NULL, 1),
                GridElement(ElementType.DATA, 1),
                GridElement(ElementType.NULL, 1),
            ]),
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.DATA, 1),
                GridElement(ElementType.NULL, 1),
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 1),
                GridElement(ElementType.NULL, 1),
            ]),
        ]
        grid_structure = [
            SymbolSection(
                num_subcarriers // 8,
                [0, 1],
                0,
            ),
        ]

        waveform = OTFSWaveform(
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            subcarrier_spacing=10e3,
            num_subcarriers=num_subcarriers,
            oversampling_factor=2,
            modulation_order=4,
        )
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()
        self.link.waveform = waveform

        # Properly configure the error correction
        self.repeater.repetitions = 3
        self.repeater.bit_block_size = waveform.bits_per_frame() // self.repeater.repetitions
        self.interleaver.enabled = False

        return waveform

    # =======================
    # Channel configurations
    # =======================

    def __configure_COST259_channel(self) -> Cost259:
        """Configure a COST259 channel with default parameters.

        Returns: The configured channel.
        """

        channel = Cost259(gain=1.0, doppler_frequency=self._doppler_frequency, num_sinusoids=100)
        return channel

    def __configure_5GTDL_channel(self) -> TDL:
        """Configure a 5GTDL channel with default parameters.

        Returns: The configured channel.
        """

        channel = TDL(gain=0.9, model_type=TDLType.B, doppler_frequency=self._doppler_frequency, rms_delay=1e-8)
        return channel

    def __configure_CDL_channel(self) -> UrbanMicrocells:
        """Configure a clustered delay line channel with default parameters.

        Returns: The configured channel.
        """

        channel = UrbanMicrocells(gain=0.9)
        return channel

    def __configure_delay_channel(self) -> RandomDelayChannel:
        """Configure a random delay channel with default parameters.

        Returns: The configured channel.
        """

        min_delay = 0.0
        max_delay = 1e-3
        channel = RandomDelayChannel((min_delay, max_delay), model_propagation_loss=True)
        return channel

    # =======================
    # Test cases
    # =======================

    def __assert_link(self) -> None:
        
        # Test bit error rates
        ber_treshold = 1e-2
        
        # Test signal powers
        transmitted_power = self.device_transmission.emerging_signals[0].power
        expected_transmit_power = self.link.power

        self.assertGreaterEqual(ber_treshold, self.ber.evaluate().artifact().to_scalar())
        # assert_array_almost_equal(expected_transmit_power * np.ones_like(transmitted_power), transmitted_power, 1, "Transmitted waveform's power does not match expectations")

    def test_ideal_channel_chirp_fsk(self) -> None:
        """Verify a valid SISO link over an ideal channel with chirp frequency shift keying modulation"""

        self.__configure_chirp_fsk_waveform()
        self.__propagate(IdealChannel(.9))
        self.__assert_link()

    def test_ideal_channel_single_carrier(self) -> None:
        """Verify a valid SISO link over an ideal channel with single carrier modulation"""

        channel = IdealChannel(.9)
        self.__configure_single_carrier_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_ideal_channel_ocdm_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OCDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = IdealChannel(.9)
        self.__configure_ocdm_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_ideal_channel_ofdm(self) -> None:
        """Verify a valid SISO link over an ideal channel OFDM modulation"""

        channel = IdealChannel(.9)
        self.__configure_ofdm_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_ideal_channel_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = IdealChannel(.9)
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        self.__propagate(channel)
        self.__assert_link()

    def test_ideal_channel_ofdm_schmidl_cox(self) -> None:
        """Verify a valid link over an AWGN channel with OFDM modluation,
        Schmidl-Cox synchronization, least-squares channel estimation and zero-forcing equalization"""

        channel = IdealChannel(.9)
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.pilot_section = SchmidlCoxPilotSection()
        waveform.synchronization = SchmidlCoxSynchronization()
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        self.__propagate(channel)
        self.__assert_link()

    def test_ideal_channel_otfs_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OTFS modulation"""

        channel = IdealChannel(.9)
        self.__configure_otfs_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_COST259_chirp_fsk(self) -> None:
        """Verify a valid SISO link over a COST259 channel with chirp frequency shift keying modulation"""

        self.__configure_chirp_fsk_waveform()
        self.__propagate(self.__configure_COST259_channel())
        self.__assert_link()

    def test_COST259_single_carrier_ideal_csi(self) -> None:
        """Verify a valid SISO link over a COST259 channel with single carrier modulation"""

        channel = self.__configure_COST259_channel()
        waveform = self.__configure_single_carrier_waveform(channel)
        waveform.guard_interval = 2e-6

        self.__propagate(channel)
        self.__assert_link()

    def test_COST259_single_carrier_ls_zf(self) -> None:
        """Verify a valid SISO link over a COST259 channel with single carrier modulation"""

        channel = self.__configure_COST259_channel()
        waveform = self.__configure_single_carrier_waveform(channel)
        waveform.guard_interval = 2e-6
        waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

        self.__propagate(channel)
        self.__assert_link()

    def test_COST259_ocdm_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OCDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = self.__configure_COST259_channel()
        self.__configure_ocdm_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_COST259_ofdm_ideal_csi(self) -> None:
        """Verify a valid SISO link over a COST259 channel with OFDM modulation"""

        channel = self.__configure_COST259_channel()
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.synchronization = Synchronization()  # Disable synchronization
        self.__propagate(channel)
        self.__assert_link()

    def test_COST259_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over a COST259 channel with OFDM modulation and least-squares channel estimation"""

        channel = self.__configure_COST259_channel()
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()

        self.__propagate(channel)
        self.__assert_link()
        
    def test_COST259_otfs_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OTFS modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = self.__configure_COST259_channel()
        self.__configure_otfs_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_5GTDL_chirp_fsk(self) -> None:
        """Verify a valid SISO link over a tapped delay line channel with chirp frequency shift keying modulation"""

        self.__configure_chirp_fsk_waveform()
        self.__propagate(self.__configure_5GTDL_channel())
        self.__assert_link()

    def test_5GTDL_channel_single_carrier(self) -> None:
        """Verify a valid SISO link over a tapped delay line channel with single carrier modulation"""

        channel = self.__configure_5GTDL_channel()
        waveform = self.__configure_single_carrier_waveform(channel)
        waveform.guard_interval = 3 * channel.rms_delay

        self.__propagate(channel)
        self.__assert_link()

    def test_5GTDL_ocdm_ls_zf(self) -> None:
        """Verify a valid SISO link over a TDL channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = self.__configure_5GTDL_channel()
        self.__configure_ofdm_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_5GTDL_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over a TDL channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = self.__configure_5GTDL_channel()
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        self.__propagate(channel)
        self.__assert_link()

    def test_5GTDL_ofdm_schmidl_cox(self) -> None:
        """Verify a valid link over a TDL channel with OFDM modluation,
        Schmidl-Cox synchronization, least-squares channel estimation and zero-forcing equalization"""
        
        channel = self.__configure_5GTDL_channel()
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.pilot_section = SchmidlCoxPilotSection()
        waveform.synchronization = SchmidlCoxSynchronization()
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()
        self.__propagate(channel)
        self.__assert_link()
        
    def test_5GTDL_otfs_ls_zf(self) -> None:
        """Verify a valid SISO link over a TDL channel with OTFS modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = self.__configure_5GTDL_channel()
        self.__configure_otfs_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_CDL_single_carrier_ideal_csi(self) -> None:
        """Verify a valid link over a clustered delay line channel with single carrier modulation and ideal CSI"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_single_carrier_waveform(channel)

        self.__propagate(channel)
        self.__assert_link()

    def test_CDL_single_carrier_ls_zf(self) -> None:
        """Verify a valid link over a clustered delay line channel with single carrier modulation"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_single_carrier_waveform(channel)
        waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

        self.__propagate(channel)
        self.__assert_link()
        
    def test_CDL_ocdm_ls_zf(self) -> None:
        """Verify a valid link over a clustered delay line channel with OCDM modulation"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_ocdm_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_CDL_ofdm_ideal_csi(self) -> None:
        """Verify a valid link over a clustered delay line channel with OFDM modulation and ideal CSI"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_ofdm_waveform(channel)

        self.__propagate(channel)
        self.__assert_link()

    def test_CDL_ofdm_ls_zf(self) -> None:
        """Verify a valid link over a clustered delay line channel with OFDM modulation"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_ofdm_waveform(channel)
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        self.__propagate(channel)
        self.__assert_link()
        
    def test_CDL_otfs_ls_zf(self) -> None:
        """Verify a valid SISO link over a CDL channel with OTFS modulation,
        least-squares channel estimation and zero-forcing equalization"""

        channel = self.__configure_CDL_channel()    
        self.__configure_otfs_waveform(channel)
        self.__propagate(channel)
        self.__assert_link()

    def test_delay_channel_ofdm_ls_zf_schmidlcox(self) -> None:
        """Verify a valid link over a delay channel with OFDM modulation"""

        channel = self.__configure_delay_channel()

        waveform = self.__configure_ofdm_waveform(channel)
        waveform.pilot_section = SchmidlCoxPilotSection()
        waveform.synchronization = SchmidlCoxSynchronization()
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        self.__propagate(channel=channel)
        self.__assert_link()


class TestSISOLinks(_TestLinksBase):
    """Test integration of simulation workflows on the link level for SISO links"""

    ...


class TestMIMOLinks(_TestLinksBase):
    """Test integration of simulation workflow on the link level"""

    def setUp(self) -> None:
        super().setUp()

        # Configure a 2x2 link scenario
        antennas = SimulatedUniformArray(SimulatedIdealAntenna(), 5e-3, [2, 1, 1])
        self.tx_device.antennas = deepcopy(antennas)
        self.rx_device.antennas = deepcopy(antennas)
        
    def test_ideal_channel_ocdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_ideal_channel_ofdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_ideal_channel_ofdm_schmidl_cox(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_ideal_channel_otfs_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_COST259_chirp_fsk(self) -> None:
        pass  # Pass since CHIRP FSK is not supported for MIMO links

    def test_COST259_single_carrier_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_COST259_ocdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_COST259_ofdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_COST259_otfs_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_5GTDL_chirp_fsk(self) -> None:
        pass  # Pass since CHIRP FSK is not supported for MIMO links

    def test_5GTDL_ocdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_5GTDL_ofdm_schmidl_cox(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_5GTDL_ofdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_5GTDL_otfs_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_CDL_ocdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_CDL_single_carrier_ls_zf(self) -> None:
        pass

    def test_CDL_ofdm_ls_zf(self) -> None:
        pass

    def test_CDL_otfs_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_delay_channel_ofdm_ls_zf_schmidlcox(self) -> None:
        pass


# Delete the base test to avoid multiple runs
del _TestLinksBase
