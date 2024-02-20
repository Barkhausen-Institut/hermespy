# -*- coding: utf-8 -*-

from .channel import (
    OrthogonalChannelEqualization,
    OrthogonalLeastSquaresChannelEstimation,
    OrthogonalZeroForcingChannelEqualization,
)
from .ocdm import OCDMWaveform
from .ofdm import (
    OFDMCorrelationSynchronization,
    OFDMWaveform,
    SchmidlCoxPilotSection,
    SchmidlCoxSynchronization,
)
from .waveform import (
    ElementType,
    GuardSection,
    GridElement,
    GridResource,
    GridSection,
    PilotSection,
    SymbolSection,
    OrthogonalWaveform,
    PrefixType,
    ReferencePosition,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "OrthogonalChannelEqualization",
    "OrthogonalLeastSquaresChannelEstimation",
    "OrthogonalZeroForcingChannelEqualization",
    "OCDMWaveform",
    "OFDMCorrelationSynchronization",
    "OFDMWaveform",
    "PilotSection",
    "SchmidlCoxPilotSection",
    "SchmidlCoxSynchronization",
    "ElementType",
    "GuardSection",
    "GridElement",
    "GridResource",
    "GridSection",
    "PilotSection",
    "SymbolSection",
    "OrthogonalWaveform",
    "PrefixType",
    "ReferencePosition",
]
