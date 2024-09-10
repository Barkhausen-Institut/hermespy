from .frame_generator import FrameGenerator
from ..bits_source import BitsSource

import numpy as np
from typing import Type

from scapy.packet import Packet, raw  # type: ignore


class FrameGeneratorScapy(FrameGenerator):
    """Scapy wrapper frame generator.

    Attrs:
       packet(Packet): Scapy packet header to which a payload would be attached.
       packet_type(Type[Packet]): Type of the first layer of the packet header.
    """

    packet: Packet
    packet_type: Type[Packet]

    def __init__(self, packet: Packet) -> None:
        """
        Args:
            packet(Packet): Packet to which a payload will be attached.
        """
        self.packet = packet
        self.packet_num_bits = len(packet)*8
        self.packet_type = packet.layers()[0]

    def pack_frame(self, source: BitsSource, num_bits: int) -> np.ndarray:
        """Generate a frame of num_bits bits from the given bitsource.
        Note that the payload size is num_bits minus number of bits in the packet header.
        Note that payload can be of size 0, in which case no data would be sent (except for the packet header).

        Args:
            source (BitsSource): payload source.
            num_bits (int): number of bits in the whole resulting frame.

        Raises:
            ValueError if num_bits is not enough to fit the packet.
        """

        payload_num_bits = num_bits - self.packet_num_bits
        if payload_num_bits < 0:
            raise ValueError(f"Packet header is bigger then the requested amount of bits ({len(self.packet)*8} > {num_bits}).")
        packet_new = self.packet_type()
        packet_new.add_payload(np.packbits(source.generate_bits(payload_num_bits)).tobytes())
        return np.unpackbits(np.frombuffer(raw(packet_new), np.uint8))

    def unpack_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.size < self.packet_num_bits:
            raise ValueError(f"The frame contains less bits then the header ({frame.size} < {self.packet_num_bits}).")
        return frame[self.packet_num_bits:]
