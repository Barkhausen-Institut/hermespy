from typing import Dict

import numpy as np

from parameters_parser.parameters_encoder import ParametersEncoder
from parameters_parser.parameters_ldpc_encoder import ParametersLdpcEncoder
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder
from parameters_parser.parameters_block_interleaver import ParametersBlockInterleaver
from parameters_parser.parameters_crc_encoder import ParametersCrcEncoder
from parameters_parser.parameters_scrambler import ParametersScrambler

from modem.coding.repetition_encoder import RepetitionEncoder
from modem.coding.ldpc_encoder import LdpcEncoder
from modem.coding.encoder import Encoder
from modem.coding.interleaver import BlockInterleaver
from modem.coding.crc_encoder import CrcEncoder
from modem.coding.scrambler import Scrambler80211a, Scrambler3GPP


class EncoderFactory:
    """Responible for creating an Encoder instance which is not contained by the factory."""

    def get_encoder(self, encoding_params: ParametersEncoder,
                          type: str, bits_in_frame: int,
                          rng: np.random.RandomState) -> Encoder:
        type = type.upper()
        self._check_validity_parameters(encoding_params, type)

        encoder: Encoder
        if type == "REPETITION":
            encoder = RepetitionEncoder(encoding_params, bits_in_frame, rng)
        elif type == "LDPC":
            encoder = LdpcEncoder(encoding_params, bits_in_frame, rng)
        elif type == "BLOCK_INTERLEAVER":
            encoder = BlockInterleaver(encoding_params, bits_in_frame, rng)
        elif type == "CRC_ENCODER":
            encoder = CrcEncoder(encoding_params, bits_in_frame, rng)
        elif type == Scrambler80211a.factory_tag:
            encoder = Scrambler80211a(encoding_params, bits_in_frame, rng)
        elif type == Scrambler3GPP.factory_tag:
            encoder = Scrambler3GPP(encoding_params, bits_in_frame, rng)
        else:
            encoder = RepetitionEncoder(ParametersRepetitionEncoder(), bits_in_frame, None)
        return encoder

    def _check_validity_parameters(self, params: ParametersEncoder, type: str) -> None:
        wrong_parameter = False
        VALID_COMBINATIONS: Dict[str, ParametersEncoder] = {
            'REPETITION': ParametersRepetitionEncoder,
            'LDPC': ParametersLdpcEncoder,
            'BLOCK_INTERLEAVER': ParametersBlockInterleaver,
            'CRC_ENCODER': ParametersCrcEncoder,
            Scrambler3GPP.factory_tag: ParametersScrambler,
            Scrambler80211a.factory_tag: ParametersScrambler
        }

        if type.upper() in VALID_COMBINATIONS.keys():
            if not isinstance(params, VALID_COMBINATIONS[type]):
                wrong_parameter = True
        elif not isinstance(params, ParametersRepetitionEncoder):
            wrong_parameter = True

        if wrong_parameter:
            raise ValueError(f"Wrong parameter type for {type}.")
