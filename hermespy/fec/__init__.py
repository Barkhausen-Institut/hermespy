from .coding import Encoder, EncoderManager
from .block_interleaver import BlockInterleaver
from .cyclic_redundancy_check import CyclicRedundancyCheck
from .repetition_encoder import RepetitionEncoder
from .scrambler import Scrambler3GPP, Scrambler80211a


# Attempt to import the C++ bindings of aff3ct.
# They are treated as an optional feature.
try:
    from .aff3ct import BCHCoding, LDPCCoding, PolarSCCoding, PolarSCLCoding, ReedSolomonCoding, RSCCoding, TurboCoding

except ModuleNotFoundError:
    ...


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"

__all__ = ["Encoder", "EncoderManager", "BlockInterleaver", "CyclicRedundancyCheck", "BCHCoding", "LDPCCoding", "PolarSCCoding", "PolarSCLCoding", "ReedSolomonCoding", "RSCCoding", "TurboCoding", "RepetitionEncoder", "Scrambler80211a", "Scrambler3GPP"]
