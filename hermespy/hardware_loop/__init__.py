# -*- coding: utf-8 -*-

from .calibration import DelayCalibration, SelectiveLeakageCalibration
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
    DelayCalibrationBase,
    LeakageCalibrationBase,
    NoDelayCalibration,
    NoLeakageCalibration,
    PhysicalDevice,
    PDT,
)
from .physical_device_dummy import PhysicalDeviceDummy, PhysicalScenarioDummy
from .scenario import PhysicalScenario, PhysicalScenarioType, SimulatedPhysicalScenario
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
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"

__all__ = [
    "DelayCalibration",
    "SelectiveLeakageCalibration",
    "EvaluatorRegistration",
    "EvaluatorPlotMode",
    "HardwareLoop",
    "HardwareLoopPlot",
    "HardwareLoopSample",
    "IterationPriority",
    "Calibration",
    "DelayCalibrationBase",
    "LeakageCalibrationBase",
    "NoDelayCalibration",
    "NoLeakageCalibration",
    "PhysicalDevice",
    "PDT",
    "PhysicalDeviceDummy",
    "PhysicalScenarioDummy",
    "PhysicalScenario",
    "PhysicalScenarioType",
    "SimulatedPhysicalScenario",
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
