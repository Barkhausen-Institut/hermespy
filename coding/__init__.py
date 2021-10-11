from __future__ import annotations
from .encoder import Encoder
from .encoder_manager import EncoderManager
from .interleaver import Interleaver
from .repetition_encoder import RepetitionEncoder

try:
    from .ldpc_binding.ldpc import LDPC

except ImportError:

    from .ldpc import LDPC
    import warnings

    warnings.warn("LDPC C++ binding could not be imported, falling back to slower Python LDPC implementation")

__all__ = [Encoder, EncoderManager, Interleaver, LDPC, RepetitionEncoder]
