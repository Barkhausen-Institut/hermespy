# -*- coding: utf-8 -*-

from .bits_source import RandomBitsSource, StreamBitsSource
from .symbols import Symbol, Symbols, StatedSymbols
from .modem import (
    CommunicationReception,
    CommunicationReceptionFrame,
    CommunicationTransmission,
    CommunicationTransmissionFrame,
    BaseModem,
    TransmittingModem,
    ReceivingModem,
    DuplexModem,
    SimplexLink,
)
from .waveform import (
    CommunicationWaveform,
    WaveformType,
    Synchronization,
    PilotCommunicationWaveform,
    PilotSymbolSequence,
    UniformPilotSymbolSequence,
    CustomPilotSymbolSequence,
    MappedPilotSymbolSequence,
    ConfigurablePilotWaveform,
    ChannelEstimation,
    ChannelEqualization,
    ZeroForcingChannelEqualization,
)
from .waveform_chirp_fsk import (
    ChirpFSKWaveform,
    ChirpFSKSynchronization,
    ChirpFSKCorrelationSynchronization,
)
from .waveform_correlation_synchronization import CorrelationSynchronization
from .waveform_single_carrier import (
    FilteredSingleCarrierWaveform,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
    SingleCarrierMinimumMeanSquareChannelEqualization,
    SingleCarrierCorrelationSynchronization,
    RaisedCosineWaveform,
    RootRaisedCosineWaveform,
    FMCWWaveform,
    RectangularWaveform,
)
from .waveforms.orthogonal import (
    OrthogonalChannelEqualization,
    OrthogonalLeastSquaresChannelEstimation,
    OrthogonalZeroForcingChannelEqualization,
    OCDMWaveform,
    OFDMCorrelationSynchronization,
    OFDMWaveform,
    PilotSection,
    SchmidlCoxPilotSection,
    SchmidlCoxSynchronization,
    ElementType,
    GridElement,
    GuardSection,
    GridResource,
    GridSection,
    SymbolSection,
    OrthogonalWaveform,
    PrefixType,
    ReferencePosition,
)

from .precoding import (
    Alamouti,
    Ganesan,
    SymbolPrecoding,
    SymbolPrecoder,
    SingleCarrier,
    SpatialMultiplexing,
    DFT,
    MaximumRatioCombining,
)
from .evaluators import (
    BitErrorEvaluator,
    BlockErrorEvaluator,
    FrameErrorEvaluator,
    ThroughputEvaluator,
)
from .tools import PskQamMapping

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Class name aliasing
SCLeastSquaresChannelEstimation = SingleCarrierLeastSquaresChannelEstimation
SCZeroForcingChannelEqualization = SingleCarrierZeroForcingChannelEqualization
SCMinimumMeanSquareChannelEqualization = SingleCarrierMinimumMeanSquareChannelEqualization
SCCorrelationSynchronization = SingleCarrierCorrelationSynchronization
RCWaveform = RaisedCosineWaveform
RRCWaveform = RootRaisedCosineWaveform
RectWaveform = RectangularWaveform


__all__ = [
    "RandomBitsSource",
    "StreamBitsSource",
    "Symbol",
    "Symbols",
    "StatedSymbols",
    "CommunicationReception",
    "CommunicationReceptionFrame",
    "CommunicationTransmission",
    "CommunicationTransmissionFrame",
    "BaseModem",
    "TransmittingModem",
    "ReceivingModem",
    "DuplexModem",
    "SimplexLink",
    "CommunicationWaveform",
    "WaveformType",
    "Synchronization",
    "PilotCommunicationWaveform",
    "PilotSymbolSequence",
    "UniformPilotSymbolSequence",
    "CustomPilotSymbolSequence",
    "MappedPilotSymbolSequence",
    "ConfigurablePilotWaveform",
    "ChannelEstimation",
    "ChannelEqualization",
    "ZeroForcingChannelEqualization",
    "ChirpFSKWaveform",
    "ChirpFSKSynchronization",
    "ChirpFSKCorrelationSynchronization",
    "CorrelationSynchronization",
    "FilteredSingleCarrierWaveform",
    "SingleCarrierLeastSquaresChannelEstimation",
    "SingleCarrierZeroForcingChannelEqualization",
    "SingleCarrierMinimumMeanSquareChannelEqualization",
    "SingleCarrierCorrelationSynchronization",
    "RaisedCosineWaveform",
    "RootRaisedCosineWaveform",
    "FMCWWaveform",
    "RectangularWaveform",
    "OFDMWaveform",
    "PilotSection",
    "SchmidlCoxPilotSection",
    "GridSection",
    "SymbolSection",
    "GuardSection",
    "GridResource",
    "GridElement",
    "ElementType",
    "PrefixType",
    "OrthogonalChannelEqualization",
    "OrthogonalLeastSquaresChannelEstimation",
    "OrthogonalZeroForcingChannelEqualization",
    "OrthogonalWaveform",
    "OCDMWaveform",
    "OFDMCorrelationSynchronization",
    "SchmidlCoxSynchronization",
    "ReferencePosition",
    "Synchronization",
    "Alamouti",
    "Ganesan",
    "SymbolPrecoding",
    "SymbolPrecoder",
    "SingleCarrier",
    "SpatialMultiplexing",
    "DFT",
    "MaximumRatioCombining",
    "BitErrorEvaluator",
    "BlockErrorEvaluator",
    "FrameErrorEvaluator",
    "ThroughputEvaluator",
    "PskQamMapping",
    "SCLeastSquaresChannelEstimation",
    "SCZeroForcingChannelEqualization",
    "SCMinimumMeanSquareChannelEqualization",
    "SCCorrelationSynchronization",
    "RCWaveform",
    "RRCWaveform",
    "RectWaveform",
]
