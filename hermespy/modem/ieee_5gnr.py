# -*- coding: utf -8 -*-

from __future__ import annotations
from os.path import abspath, dirname, join
from typing import Sequence, Type
from typing_extensions import override

import numpy as np

# Attempt import LDPC channel coding. Might not be available on Windows
try:
    from hermespy.fec import LDPCCoding  # type: ignore
except ImportError:
    LDPCCoding = None  # type: ignore
from hermespy.core import DeserializationProcess
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

    __slot: NRSlot

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
        self.__slot = NRSlot(num_resource_blocks=num_resource_blocks)

        # Least-squares channel estimation and zero-forcing equalization are used by default
        self.__slot.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        self.__slot.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        # Initialize link with the given parameters
        SimplexLink.__init__(
            self,
            selected_transmit_ports=selected_transmit_ports,
            selected_receive_ports=selected_receive_ports,
            carrier_frequency=carrier_frequency,
            bits_source=bits_source,
            frame_generator=frame_generator,
            waveform=self.__slot,
            seed=seed,
        )

        # Configure an LDPC code with rate R=1/2 and block length 128
        # Note that this is not standard-compliant
        if LDPCCoding:
            ldpc_code = join(dirname(abspath(__file__)), 'resources', 'ofdm_ldpc.alist')
            self.encoder_manager.add_encoder(LDPCCoding(100, ldpc_code, "", True, 10))

    @override
    @classmethod
    def Deserialize(cls: Type[NRSlotLink], process: DeserializationProcess) -> NRSlotLink:
        slot = process.deserialize_object("waveform", NRSlot)
        selected_transmit_ports = process.deserialize_array("selected_transmit_ports", np.int64, None)
        selected_receive_ports = process.deserialize_array("selected_receive_ports", np.int64, None)
        return NRSlotLink(
            num_resource_blocks=slot.num_resource_blocks,
            selected_transmit_ports=selected_transmit_ports.tolist() if selected_transmit_ports else None,
            selected_receive_ports=selected_receive_ports.tolist() if selected_receive_ports else None,
            carrier_frequency=process.deserialize_floating("carrier_frequency", None),
            bits_source=process.deserialize_object("bits_source", BitsSource),
            frame_generator=process.deserialize_object("frame_generator", FrameGenerator),
            seed=process.deserialize_integer("seed", None),
        )
