# -*- coding: utf-8 -*-

from .calibration import DelayCalibration, ScalarAntennaCalibration, SelectiveLeakageCalibration
from .hardware_loop import (
    EvaluatorRegistration,
    EvaluatorPlotMode,
    HardwareLoop,
    HardwareLoopPlot,
    HardwareLoopSample,
    IterationPriority,
)
from .physical_device import (
    Calibration,
    AntennaCalibration,
    DelayCalibrationBase,
    LeakageCalibrationBase,
    NoAntennaCalibration,
    NoDelayCalibration,
    NoLeakageCalibration,
    PhysicalDevice,
    PhysicalDeviceState,
    PDT,
)
from .precoding import IQCombiner, IQSplitter
from .physical_device_dummy import PhysicalDeviceDummy, PhysicalScenarioDummy
from .scenario import PhysicalScenario, PhysicalScenarioType
from .audio import AudioDevice, AudioScenario
from .visualizers import (
    DeviceReceptionPlot,
    DeviceTransmissionPlot,
    EyePlot,
    ReceivedConstellationPlot,
    RadarRangePlot,
    EvaluationPlot,
    ArtifactPlot,
)

try:  # pragma: no cover
    from .uhd import UsrpAntennas, UsrpDevice, UsrpSystem
except ImportError:  # pragma: no cover
    UsrpAntennas, UsrpDevice, UsrpSystem = None, None, None  # type: ignore

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"

__all__ = [
    "ScalarAntennaCalibration",
    "DelayCalibration",
    "SelectiveLeakageCalibration",
    "EvaluatorRegistration",
    "EvaluatorPlotMode",
    "HardwareLoop",
    "HardwareLoopPlot",
    "HardwareLoopSample",
    "IterationPriority",
    "Calibration",
    "AntennaCalibration",
    "DelayCalibrationBase",
    "LeakageCalibrationBase",
    "NoAntennaCalibration",
    "NoDelayCalibration",
    "NoLeakageCalibration",
    "PhysicalDevice",
    "PhysicalDeviceState",
    "PDT",
    "IQCombiner",
    "IQSplitter",
    "PhysicalDeviceDummy",
    "PhysicalScenarioDummy",
    "PhysicalScenario",
    "PhysicalScenarioType",
    "UsrpAntennas",
    "UsrpDevice",
    "UsrpSystem",
    "AudioDevice",
    "AudioScenario",
    "DeviceReceptionPlot",
    "DeviceTransmissionPlot",
    "EyePlot",
    "ReceivedConstellationPlot",
    "RadarRangePlot",
    "EvaluationPlot",
    "ArtifactPlot",
]
