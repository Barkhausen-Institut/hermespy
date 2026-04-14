# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from hermespy.core import Serializable, SerializationProcess, DeserializationProcess
from .aff3ct import (
    BCHCoding as _BCHCoding,
    LDPCCoding as _LDPCCoding,
    PolarSCCoding as _PolarSCCoding,
    PolarSCLCoding as _PolarSCLCoding,
    ReedSolomonCoding as _ReedSolomonCoding,
    RSCCoding as _RSCCoding,
    TurboCoding as _TurboCoding,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class LDPCCoding(_LDPCCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, str, str, bool, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'num_iterations')
        process.serialize_string(state[1], 'h_path')
        process.serialize_string(state[2], 'g_path')
        process.serialize_integer(int(state[3]), 'syndrome_checking')
        process.serialize_integer(state[4], 'min_num_iterations')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> LDPCCoding:
        return LDPCCoding(
            process.deserialize_integer('num_iterations'),
            process.deserialize_string('h_path'),
            process.deserialize_string('g_path'),
            bool(process.deserialize_integer('syndrome_checking')),
            process.deserialize_integer('min_num_iterations'),
        )


class BCHCoding(_BCHCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, int, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'data_block_size')
        process.serialize_integer(state[1], 'code_block_size')
        process.serialize_integer(state[2], 'power')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> BCHCoding:
        return BCHCoding(
            process.deserialize_integer('data_block_size'),
            process.deserialize_integer('code_block_size'),
            process.deserialize_integer('power'),
        )


class PolarSCCoding(_PolarSCCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, int, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'data_block_size')
        process.serialize_integer(state[1], 'code_block_size')
        process.serialize_integer(state[2], 'ber')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> PolarSCCoding:
        return PolarSCCoding(
            process.deserialize_integer('data_block_size'),
            process.deserialize_integer('code_block_size'),
            process.deserialize_integer('ber'),
        )


class PolarSCLCoding(_PolarSCLCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, int, int, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'data_block_size')
        process.serialize_integer(state[1], 'code_block_size')
        process.serialize_integer(state[2], 'ber')
        process.serialize_integer(state[3], 'num_paths')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> PolarSCLCoding:
        return PolarSCLCoding(
            process.deserialize_integer('data_block_size'),
            process.deserialize_integer('code_block_size'),
            process.deserialize_integer('ber'),
            process.deserialize_integer('num_paths'),
        )


class ReedSolomonCoding(_ReedSolomonCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'data_block_size')
        process.serialize_integer(state[1], 'correction_power')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> ReedSolomonCoding:
        return ReedSolomonCoding(
            process.deserialize_integer('data_block_size'),
            process.deserialize_integer('correction_power'),
        )


class RSCCoding(_RSCCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, int, bool, int, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'bit_block_size')
        process.serialize_integer(state[1], 'code_block_size')
        process.serialize_integer(int(state[2]), 'buffered_coding')
        process.serialize_integer(state[3], 'poly_a')
        process.serialize_integer(state[4], 'poly_b')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> RSCCoding:
        return RSCCoding(
            process.deserialize_integer('bit_block_size'),
            process.deserialize_integer('code_block_size'),
            bool(process.deserialize_integer('buffered_coding')),
            process.deserialize_integer('poly_a'),
            process.deserialize_integer('poly_b'),
        )


class TurboCoding(_TurboCoding, Serializable):

    @override
    def serialize(self, process: SerializationProcess) -> None:
        state: tuple[int, int, int, int] = self.__getstate__()  # type: ignore
        process.serialize_integer(state[0], 'data_block_size')
        process.serialize_integer(state[1], 'poly_a')
        process.serialize_integer(state[2], 'poly_b')
        process.serialize_integer(state[3], 'num_iterations')

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> TurboCoding:
        return TurboCoding(
            process.deserialize_integer('data_block_size'),
            process.deserialize_integer('poly_a'),
            process.deserialize_integer('poly_b'),
            process.deserialize_integer('num_iterations'),
        )
