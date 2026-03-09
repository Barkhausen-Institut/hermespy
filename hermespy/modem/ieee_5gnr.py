# -*- coding: utf -8 -*-

from os.path import abspath, dirname, join
from typing import Sequence

from hermespy.fec import LDPCCoding  # type: ignore
from .bits_source import BitsSource
from .modem import SimplexLink
from .frame_generator import FrameGenerator
from .waveforms.ieee_5gnr import IEEE_5GNR_MIN_NUM_RBS, NRSlot
from .waveforms.orthogonal import OrthogonalLeastSquaresChannelEstimation, OrthogonalZeroForcingChannelEqualization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NRSlotLink(SimplexLink):
    """
    A simplex link for 5G NR simulations considering only a single slot.
    """

    def __init__(
        self,
        num_resource_blocks: int = IEEE_5GNR_MIN_NUM_RBS,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        carrier_frequency: float | None = None,
        bits_source: BitsSource | None = None,
        frame_generator: FrameGenerator | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            num_resource_blocks:
                Number of resource blocks within a single slot.
                Must be at least 24 to meet the minimum slot bandwidth requirements of 5G NR.
                The maximum number depends on the overall bandwidth available for the given frequency range.

            selected_transmit_ports:
                The indices of the transmit ports to be used.
                If :py:obj:`None`, all available ports are used.

            selected_receive_ports:
                The indices of the receive ports to be used.
                If :py:obj:`None`, all available ports are used.

            carrier_frequency:
                The carrier frequency in Hz.
                If :py:obj:`None`, the default carrier frequency of the devices is used.

            bits_source:
                The bits source to be used for generating the transmitted bits.
                If :py:obj:`None`, a default random bits source is used.

            frame_generator:
                The frame generator to be used for generating the frames.
                If :py:obj:`None`, a default frame generator is used.

            seed:
                The seed for the random number generator.
                If :py:obj:`None`, a random seed is used.
        """

        # Initialize waveform
        slot = NRSlot(num_resource_blocks=num_resource_blocks)

        # Least-squares channel estimation and zero-forcing equalization are used by default
        slot.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        slot.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        # Initialize link with the given parameters
        SimplexLink.__init__(
            self,
            selected_transmit_ports=selected_transmit_ports,
            selected_receive_ports=selected_receive_ports,
            carrier_frequency=carrier_frequency,
            bits_source=bits_source,
            frame_generator=frame_generator,
            waveform=slot,
            seed=seed,
        )

        # Configure an LDPC code with rate R=1/2 and block length 128
        # Note that this is not standard-compliant
        ldpc_code = join(dirname(abspath(__file__)), 'resources', 'ofdm_ldpc.alist')
        self.encoder_manager.add_encoder(LDPCCoding(100, ldpc_code, "", True, 10))
