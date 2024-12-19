# -*- coding: utf-8 -*-

from .bits_source import RandomBitsSource, StreamBitsSource
from .frame_generator import FrameGenerator, FrameGeneratorStub
from .symbols import Symbol, Symbols, StatedSymbols
from .modem import (
    CommunicationReception,
    CommunicationReceptionFrame,
    CommunicationTransmission,
    CommunicationTransmissionFrame,
    BaseModem,
    TransmittingModem,
    TransmittingModemBase,
    ReceivingModem,
    ReceivingModemBase,
    DuplexModem,
    SimplexLink,
)
from .waveform import (
    CommunicationWaveform,
    CWT,
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
    OTFSWaveform,
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
    TransmitSymbolCoding,
    TransmitSymbolEncoder,
    ReceiveSymbolCoding,
    ReceiveSymbolDecoder,
    SingleCarrier,
    DFT,
    MaximumRatioCombining,
)
from .evaluators import (
    BitErrorEvaluator,
    BlockErrorEvaluator,
    FrameErrorEvaluator,
    ThroughputEvaluator,
    ConstellationEVM,
)
from .tools import PskQamMapping

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
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
    "FrameGenerator",
    "FrameGeneratorStub",
    "Symbol",
    "Symbols",
    "StatedSymbols",
    "CommunicationReception",
    "CommunicationReceptionFrame",
    "CommunicationTransmission",
    "CommunicationTransmissionFrame",
    "BaseModem",
    "TransmittingModem",
    "TransmittingModemBase",
    "ReceivingModem",
    "ReceivingModemBase",
    "DuplexModem",
    "SimplexLink",
    "CommunicationWaveform",
    "CWT",
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
    "OTFSWaveform",
    "OFDMCorrelationSynchronization",
    "SchmidlCoxSynchronization",
    "ReferencePosition",
    "Synchronization",
    "OTFSWaveform",
    "Alamouti",
    "Ganesan",
    "TransmitSymbolCoding",
    "TransmitSymbolEncoder",
    "ReceiveSymbolCoding",
    "ReceiveSymbolDecoder",
    "SymbolPrecoder",
    "SingleCarrier",
    "DFT",
    "MaximumRatioCombining",
    "BitErrorEvaluator",
    "BlockErrorEvaluator",
    "FrameErrorEvaluator",
    "ThroughputEvaluator",
    "ConstellationEVM",
    "PskQamMapping",
    "SCLeastSquaresChannelEstimation",
    "SCZeroForcingChannelEqualization",
    "SCMinimumMeanSquareChannelEqualization",
    "SCCorrelationSynchronization",
    "RCWaveform",
    "RRCWaveform",
    "RectWaveform",
]
