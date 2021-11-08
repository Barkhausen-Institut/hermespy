from __future__ import annotations
from .encoder import Encoder
from .encoder_manager import EncoderManager
from .block_interleaver import BlockInterleaver
from .repetition_encoder import RepetitionEncoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


try:
    from .ldpc_binding.ldpc import LDPC

except ImportError:

    from .ldpc import LDPC
    import warnings

    warnings.warn("LDPC C++ binding could not be imported, falling back to slower Python LDPC implementation")

__all__ = ['Encoder', 'EncoderManager', 'BlockInterleaver', 'LDPC', 'RepetitionEncoder']
