from .cube import RadarCube
from .detection import (
    RadarDetector,
    PointDetection,
    RadarPointCloud,
    ThresholdDetector,
    MaxDetector,
    CFARDetector,
)
from .radar import Radar, RadarBase, RadarWaveform, RadarTransmission, RadarReception
from .fmcw import FMCW
from .evaluators import ReceiverOperatingCharacteristic, DetectionProbEvaluator, RootMeanSquareError

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"

__all__ = [
    "RadarCube",
    "RadarDetector",
    "PointDetection",
    "RadarPointCloud",
    "ThresholdDetector",
    "MaxDetector",
    "CFARDetector",
    "Radar",
    "RadarBase",
    "RadarWaveform",
    "RadarTransmission",
    "RadarReception",
    "FMCW",
    "ReceiverOperatingCharacteristic",
    "DetectionProbEvaluator",
    "RootMeanSquareError",
]
