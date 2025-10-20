# -*- coding: utf-8 -*-

from .ad import ADC, DAC, Gain, AutomaticGainControl, QuantizerType, GainControlType
from .amps import (
    PowerAmplifier,
    SalehPowerAmplifier,
    RappPowerAmplifier,
    ClippingPowerAmplifier,
    CustomPowerAmplifier,
)
from .filters import HPF
from .mixers import MixerType, IdealMixer, Mixer
from .ramp import RampGenerator
from .shift import Shift
from .source import Source
from .split import Split
from .sum import Sum
from .x4 import X4

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
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
]
