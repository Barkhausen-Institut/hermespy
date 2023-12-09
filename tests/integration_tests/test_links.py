# -*- coding: utf-8 -*-

from copy import deepcopy
from unittest import TestCase

import numpy as np

from hermespy.channel import Channel, IdealChannel, MultipathFading5GTDL, MultipathFadingCost259, RandomDelayChannel, TDLType, StreetCanyonOutsideToInside
from hermespy.core import Transformation
from hermespy.simulation import SimulatedDevice, SimulationScenario, SingleCarrierIdealChannelEstimation, OFDMIdealChannelEstimation, SimulatedUniformArray, SimulatedIdealAntenna
from hermespy.modem import (
    SpatialMultiplexing,
    SimplexLink,
    BitErrorEvaluator,
    RootRaisedCosineWaveform,
    SingleCarrierCorrelationSynchronization,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
    ChirpFSKWaveform,
    ChirpFSKCorrelationSynchronization,
    OFDMWaveform,
    FrameResource,
    FrameSymbolSection,
    FrameElement,
    ElementType,
    OFDMCorrelationSynchronization,
    PilotSection,
    OFDMLeastSquaresChannelEstimation,
    OFDMZeroForcingChannelEqualization,
    SchmidlCoxPilotSection,
    SchmidlCoxSynchronization,
    ReferencePosition,
)
from hermespy.fec import RepetitionEncoder, BlockInterleaver

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestLinksBase(TestCase):
    """Base class for link integration tests."""

    tx_device: SimulatedDevice
    rx_device: SimulatedDevice
    ber: BitErrorEvaluator
    link: SimplexLink
    _doppler_frequency: float = 100.0

    def setUp(self) -> None:
        # Configure a 1x1link scenario
        scenario = SimulationScenario(seed=42)
        carrier_frequency = 2.4e9
        self.tx_device = scenario.new_device(pose=Transformation.From_Translation(np.array([0, 0, 50])), carrier_frequency=carrier_frequency)
        self.rx_device = scenario.new_device(pose=Transformation.From_Translation(np.array([80, 80, 20])), carrier_frequency=carrier_frequency)
        self.rx_device.velocity = 1 * (self.rx_device.position - self.tx_device.position) / np.linalg.norm(self.rx_device.position - self.tx_device.position)

        # Define a simplex linke between the two devices
        self.link = SimplexLink(self.tx_device, self.rx_device)
        self.link.precoding[0] = SpatialMultiplexing()
        # self.link.precoding[1] = DFT()

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

        device_transmission = self.tx_device.transmit()
        channel_realization = channel.realize()
        channel_propagation = channel_realization.propagate(device_transmission)
        self.rx_device.process_input(channel_propagation)
        link_reception = self.link.receive()

        # Debug:
        #
        # link_transmission = device_transmission.operator_transmissions[0]
        # link_reception = self.link.receive()
        # tx = link_transmission.signal
        # state = channel_realization.state(0, tx.sampling_rate, tx.num_samples, tx.num_samples)
        # rx_prediction = state.propagate(tx)
        return

    # =======================
    # Waveform configurations
    # =======================

    def __configure_single_carrier_waveform(self) -> RootRaisedCosineWaveform:
        """Configure a single carrier wafeform with default parameters.

        Returns: The configured waveform.
        """

        waveform = RootRaisedCosineWaveform(symbol_rate=1 / 10e-6, num_preamble_symbols=10, num_data_symbols=40, pilot_rate=10, oversampling_factor=8, roll_off=0.9)
        waveform.synchronization = SingleCarrierCorrelationSynchronization()
        waveform.channel_estimation = SingleCarrierIdealChannelEstimation(self.tx_device, self.rx_device)
        waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
        self.link.waveform_generator = waveform

        return waveform

    def __configure_chirp_fsk_waveform(self) -> ChirpFSKWaveform:
        """Configure a chirp frequency shift keying wafeform with default parameters.

        Returns: The configured waveform.
        """

        waveform = ChirpFSKWaveform(chirp_duration=1e-5, chirp_bandwidth=375e6)
        waveform.synchronization = ChirpFSKCorrelationSynchronization()
        self.link.waveform_generator = waveform

        return waveform

    def __configure_ofdm_waveform(self) -> OFDMWaveform:
        """Configure an OFDM wafeform with default parameters.

        Returns: The configured waveform.
        """

        # Mock 5G numerology #1:
        # 120khz subcarrier spacing, 120 subcarriers, 2us guard interval, 1ms subframe duration

        num_symbols = 15

        resources = [FrameResource(200, prefix_ratio=0.0684, elements=[FrameElement(ElementType.REFERENCE, 1), FrameElement(ElementType.DATA, 4)]), FrameResource(1000, prefix_ratio=0.0684, elements=[FrameElement(ElementType.DATA, 1)])]
        structure = [FrameSymbolSection(num_symbols // 3, [0, 1, 1, 1, 1])]

        waveform = OFDMWaveform(subcarrier_spacing=3.75e3, num_subcarriers=1000, dc_suppression=True, resources=resources, structure=structure)
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OFDMIdealChannelEstimation(self.tx_device, self.rx_device, reference_position=ReferencePosition.IDEAL)
        waveform.channel_equalization = OFDMZeroForcingChannelEqualization()

        self.link.waveform_generator = waveform

        # Properly configure the error correction
        bits_per_symbol = waveform.bits_per_frame() // num_symbols
        self.repeater.bit_block_size = bits_per_symbol // self.repeater.repetitions
        self.interleaver.block_size = bits_per_symbol
        self.interleaver.interleave_blocks = waveform.bits_per_symbol

        # Debugging: Deactivate error correction
        self.repeater.enabled = False
        self.interleaver.enabled = False

        return waveform

    # =======================
    # Channel configurations
    # =======================

    def __configure_COST259_channel(self) -> MultipathFadingCost259:
        """Configure a COST259 channel with default parameters.

        Returns: The configured channel.
        """

        channel = MultipathFadingCost259(alpha_device=self.tx_device, beta_device=self.rx_device, gain=0.9, doppler_frequency=self._doppler_frequency)
        return channel

    def __configure_5GTDL_channel(self) -> MultipathFading5GTDL:
        """Configure a 5GTDL channel with default parameters.

        Returns: The configured channel.
        """

        channel = MultipathFading5GTDL(alpha_device=self.tx_device, beta_device=self.rx_device, gain=0.9, model_type=TDLType.B, doppler_frequency=self._doppler_frequency, rms_delay=1e-8)
        return channel

    def __configure_CDL_channel(self) -> StreetCanyonOutsideToInside:
        """Configure a clustered delay line channel with default parameters.

        Returns: The configured channel.
        """

        channel = StreetCanyonOutsideToInside(self.tx_device, self.rx_device, 0.9)
        return channel

    def __configure_delay_channel(self) -> RandomDelayChannel:
        """Configure a random delay channel with default parameters.

        Returns: The configured channel.
        """

        min_delay = 0.0
        max_delay = 1e-3
        channel = RandomDelayChannel((min_delay, max_delay), alpha_device=self.tx_device, beta_device=self.rx_device, model_propagation_loss=True)
        return channel

    # =======================
    # Test cases
    # =======================

    def __assert_link(self) -> None:
        ber_treshold = 1e-2
        self.assertGreaterEqual(ber_treshold, self.ber.evaluate().artifact().to_scalar())

    def test_ideal_channel_chirp_fsk(self) -> None:
        """Verify a valid SISO link over an ideal channel with chirp frequency shift keying modulation"""

        self.__configure_chirp_fsk_waveform()
        self.__propagate(IdealChannel(self.tx_device, self.rx_device))
        self.__assert_link()

    def test_ideal_channel_single_carrier(self) -> None:
        """Verify a valid SISO link over an ideal channel with single carrier modulation"""

        self.__configure_single_carrier_waveform()
        self.__propagate(IdealChannel(self.tx_device, self.rx_device))
        self.__assert_link()

    def test_ideal_channel_ofdm(self) -> None:
        """Verify a valid SISO link over an ideal channel ofdm modulation"""

        self.__configure_ofdm_waveform()
        self.__propagate(IdealChannel(self.tx_device, self.rx_device))
        self.__assert_link()

    def test_ideal_channel_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over an ideal channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        waveform = self.__configure_ofdm_waveform()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        waveform.channel_equalization = OFDMZeroForcingChannelEqualization()

        self.__propagate(IdealChannel(self.tx_device, self.rx_device))
        self.__assert_link()

    def test_ideal_channel_ofdm_schmidl_cox(self) -> None:
        """Verify a valid link over an AWGN channel with OFDM modluation,
        Schmidl-Cox synchronization, least-squares channel estimation and zero-forcing equalization"""

        waveform = self.__configure_ofdm_waveform()
        waveform.pilot_section = SchmidlCoxPilotSection()
        waveform.synchronization = SchmidlCoxSynchronization()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        waveform.channel_equalization = OFDMZeroForcingChannelEqualization()

        self.__propagate(IdealChannel(self.tx_device, self.rx_device))
        self.__assert_link()

    def test_COST259_chirp_fsk(self) -> None:
        """Verify a valid SISO link over a COST259 channel with chirp frequency shift keying modulation"""

        self.__configure_chirp_fsk_waveform()
        self.__propagate(self.__configure_COST259_channel())
        self.__assert_link()

    def test_COST259_single_carrier_ideal_csi(self) -> None:
        """Verify a valid SISO link over a COST259 channel with single carrier modulation"""

        waveform = self.__configure_single_carrier_waveform()
        waveform.guard_interval = 2e-6

        self.__propagate(self.__configure_COST259_channel())
        self.__assert_link()

    def test_COST259_single_carrier_ls_zf(self) -> None:
        """Verify a valid SISO link over a COST259 channel with single carrier modulation"""

        channel = self.__configure_COST259_channel()
        waveform = self.__configure_single_carrier_waveform()
        waveform.guard_interval = 2e-6
        waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

        self.__propagate(channel=channel)
        self.__assert_link()

    def test_COST259_ofdm_ideal_csi(self) -> None:
        """Verify a valid SISO link over a COST259 channel with OFDM modulation"""

        self.__configure_ofdm_waveform()
        self.__propagate(self.__configure_COST259_channel())
        self.__assert_link()

    def test_COST259_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over a COST259 channel with OFDM modulation and least-squares channel estimation"""

        waveform = self.__configure_ofdm_waveform()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()

        self.__propagate(self.__configure_COST259_channel())
        self.__assert_link()

    def test_5GTDL_chirp_fsk(self) -> None:
        """Verify a valid SISO link over a tapped delay line channel with chirp frequency shift keying modulation"""

        self.__configure_chirp_fsk_waveform()
        self.__propagate(self.__configure_5GTDL_channel())
        self.__assert_link()

    def test_5GTDL_channel_single_carrier(self) -> None:
        """Verify a valid SISO link over a tapped delay line channel with single carrier modulation"""

        channel = self.__configure_5GTDL_channel()
        waveform = self.__configure_single_carrier_waveform()
        waveform.guard_interval = 3 * channel.rms_delay

        self.__propagate(channel=channel)
        self.__assert_link()

    def test_5GTDL_ofdm_ls_zf(self) -> None:
        """Verify a valid SISO link over a TDL channel with OFDM modulation,
        least-squares channel estimation and zero-forcing equalization"""

        waveform = self.__configure_ofdm_waveform()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        waveform.channel_equalization = OFDMZeroForcingChannelEqualization()

        self.__propagate(self.__configure_5GTDL_channel())
        self.__assert_link()

    def test_5GTDL_ofdm_schmidl_cox(self) -> None:
        """Verify a valid link over a TDL channel with OFDM modluation,
        Schmidl-Cox synchronization, least-squares channel estimation and zero-forcing equalization"""

        waveform = self.__configure_ofdm_waveform()
        waveform.pilot_section = SchmidlCoxPilotSection()
        waveform.synchronization = SchmidlCoxSynchronization()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        waveform.channel_equalization = OFDMZeroForcingChannelEqualization()

        self.__propagate(self.__configure_5GTDL_channel())
        self.__assert_link()

    def test_CDL_single_carrier_ideal_csi(self) -> None:
        """Verify a valid link over a clustered delay line channel with single carrier modulation and ideal CSI"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_single_carrier_waveform()

        self.__propagate(channel=channel)
        self.__assert_link()

    def test_CDL_single_carrier_ls_zf(self) -> None:
        """Verify a valid link over a clustered delay line channel with single carrier modulation"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_single_carrier_waveform()
        waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

        self.__propagate(channel=channel)
        self.__assert_link()

    def test_CDL_ofdm_ideal_csi(self) -> None:
        """Verify a valid link over a clustered delay line channel with OFDM modulation and ideal CSI"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_ofdm_waveform()

        self.__propagate(channel=channel)
        self.__assert_link()

    def test_CDL_ofdm_ls_zf(self) -> None:
        """Verify a valid link over a clustered delay line channel with OFDM modulation"""

        channel = self.__configure_CDL_channel()
        waveform = self.__configure_ofdm_waveform()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        self.__propagate(channel=channel)
        self.__assert_link()

    def test_delay_channel_ofdm_ls_zf_schmidlcox(self) -> None:
        """Verify a valid link over a delay channel with OFDM modulation"""

        channel = self.__configure_delay_channel()

        waveform = self.__configure_ofdm_waveform()
        waveform.pilot_section = SchmidlCoxPilotSection()
        waveform.synchronization = SchmidlCoxSynchronization()
        waveform.channel_estimation = OFDMLeastSquaresChannelEstimation()
        waveform.channel_equalization = OFDMZeroForcingChannelEqualization()

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

    def test_ideal_channel_ofdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_ideal_channel_ofdm_schmidl_cox(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_COST259_chirp_fsk(self) -> None:
        pass  # Pass since CHIRP FSK is not supported for MIMO links

    def test_COST259_single_carrier_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_COST259_ofdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_5GTDL_chirp_fsk(self) -> None:
        pass  # Pass since CHIRP FSK is not supported for MIMO links

    def test_5GTDL_ofdm_schmidl_cox(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_5GTDL_ofdm_ls_zf(self) -> None:
        pass  # Pass the test since least-squares channel estimation is not supported for MIMO links

    def test_CDL_single_carrier_ls_zf(self) -> None:
        pass

    def test_CDL_ofdm_ls_zf(self) -> None:
        pass

    def test_delay_channel_ofdm_ls_zf_schmidlcox(self) -> None:
        pass


# Delete the base test to avoid multiple runs
del _TestLinksBase
