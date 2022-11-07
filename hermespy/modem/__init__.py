from .bits_source import RandomBitsSource, StreamBitsSource
from .symbols import Symbol, Symbols, StatedSymbols
from .modem import CommunicationReception, CommunicationReceptionFrame, CommunicationTransmission, CommunicationTransmissionFrame, BaseModem, TransmittingModem, ReceivingModem, DuplexModem, SimplexLink
from .waveform_generator import WaveformGenerator, Synchronization, PilotWaveformGenerator, PilotSymbolSequence,  UniformPilotSymbolSequence, CustomPilotSymbolSequence, MappedPilotSymbolSequence, ConfigurablePilotWaveform, ChannelEqualization, ZeroForcingChannelEqualization
from .waveform_generator_chirp_fsk import ChirpFSKWaveform, ChirpFSKSynchronization,\
    ChirpFSKCorrelationSynchronization
from .waveform_single_carrier import FilteredSingleCarrierWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierMinimumMeanSquareChannelEqualization, SingleCarrierCorrelationSynchronization, RaisedCosineWaveform, RootRaisedCosineWaveform, FMCWWaveform, RectangularWaveform, SingleCarrierIdealChannelEstimation
from .waveform_generator_ofdm import OFDMWaveform, FrameGuardSection, FrameSymbolSection, FrameResource, PilotSection, SchmidlCoxPilotSection, FrameElement, ElementType, OFDMCorrelationSynchronization, SchmidlCoxSynchronization, OFDMMinimumMeanSquareChannelEqualization, OFDMZeroForcingChannelEqualization, OFDMIdealChannelEstimation, OFDMLeastSquaresChannelEstimation
from .evaluators import BitErrorEvaluator, BlockErrorEvaluator, FrameErrorEvaluator, ThroughputEvaluator
from .tools import PskQamMapping

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
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
SCIdealChannelEstimation = SingleCarrierIdealChannelEstimation


__all__ = [
    'RandomBitsSource', 'StreamBitsSource',
    'Symbol', 'Symbols', 'StatedSymbols',
    'CommunicationReception', 'CommunicationReceptionFrame', 'CommunicationTransmission', 'CommunicationTransmissionFrame', 'BaseModem', 'TransmittingModem', 'ReceivingModem', 'DuplexModem', 'SimplexLink',
    'WaveformGenerator', 'Synchronization', 'PilotWaveformGenerator', 'PilotSymbolSequence',  'UniformPilotSymbolSequence', 'CustomPilotSymbolSequence', 'MappedPilotSymbolSequence', 'ConfigurablePilotWaveform', 'ChannelEqualization', 'ZeroForcingChannelEqualization',
    'ChirpFSKWaveform', 'ChirpFSKSynchronization', 'ChirpFSKCorrelationSynchronization',
    'FilteredSingleCarrierWaveform', 'SingleCarrierLeastSquaresChannelEstimation', 'SingleCarrierZeroForcingChannelEqualization', 'SingleCarrierMinimumMeanSquareChannelEqualization', 'SingleCarrierCorrelationSynchronization', 'RaisedCosineWaveform', 'RootRaisedCosineWaveform', 'FMCWWaveform', 'RectangularWaveform', 'SingleCarrierIdealChannelEstimation',
    'OFDMWaveform', 'PilotSection', 'SchmidlCoxPilotSection', 'FrameSymbolSection', 'FrameGuardSection', 'FrameResource', 'FrameElement', 'ElementType', 'OFDMCorrelationSynchronization', 'SchmidlCoxSynchronization', 'OFDMMinimumMeanSquareChannelEqualization', 'OFDMZeroForcingChannelEqualization', 'OFDMIdealChannelEstimation', 'OFDMLeastSquaresChannelEstimation',
    'Synchronization', 'BitErrorEvaluator', 'BlockErrorEvaluator',
    'FrameErrorEvaluator', 'ThroughputEvaluator',
    'PskQamMapping',
    'SCLeastSquaresChannelEstimation', 'SCZeroForcingChannelEqualization', 'SCMinimumMeanSquareChannelEqualization', 'SCCorrelationSynchronization', 'RCWaveform', 'RRCWaveform', 'RectWaveform', 'SCIdealChannelEstimation',
]
