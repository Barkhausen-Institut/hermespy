# -*- coding: utf-8 -*-

from .blocks import (
    ADC,
    DAC,
    Gain,
    AutomaticGainControl,
    QuantizerType,
    GainControlType,
    PowerAmplifier,
    SalehPowerAmplifier,
    RappPowerAmplifier,
    ClippingPowerAmplifier,
    CustomPowerAmplifier,
    HPF,
    MixerType,
    IdealMixer,
    Mixer,
    RampGenerator,
    Shift,
    Source,
    Split,
    Sum,
    X4,
)
from .noise import (
    NoiseLevel,
    N0,
    ThermalNoise,
    SNR,
    NoiseModel,
    NoiseRealization,
    AWGN,
    AWGNRealization,
    PhaseNoise,
    PhaseNoiseRealization,
    NoPhaseNoise,
    OscillatorPhaseNoise,
)
from .block import RFBlock, RFBlockRealization, RFBlockPort, RFBlockPortType
from .chain import RFChain, RFChainRealization, RFBlockReference
from .signal import RFSignal

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "ADC",
    "DAC",
    "Gain",
    "AutomaticGainControl",
    "QuantizerType",
    "GainControlType",
    "PowerAmplifier",
    "SalehPowerAmplifier",
    "RappPowerAmplifier",
    "ClippingPowerAmplifier",
    "CustomPowerAmplifier",
    "HPF",
    "MixerType",
    "IdealMixer",
    "Mixer",
    "RampGenerator",
    "Shift",
    "Source",
    "Split",
    "Sum",
    "X4",
    "NoiseLevel",
    "N0",
    "ThermalNoise",
    "SNR",
    "NoiseModel",
    "NoiseRealization",
    "AWGN",
    "AWGNRealization",
    "PhaseNoise",
    "PhaseNoiseRealization",
    "NoPhaseNoise",
    "OscillatorPhaseNoise",
    "RFBlock",
    "RFBlockRealization",
    "RFBlockPort",
    "RFBlockPortType",
    "RFChain",
    "RFChainRealization",
    "RFBlockReference",
    "RFSignal",
]
