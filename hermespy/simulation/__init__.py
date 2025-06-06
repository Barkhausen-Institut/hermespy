# -*- coding: utf-8 -*-
from .animation import Trajectory, LinearTrajectory, StaticTrajectory
from .antennas import (
    SimulatedAntenna,
    SimulatedAntennaPort,
    SimulatedCustomArray,
    SimulatedAntennaArray,
    SimulatedDipole,
    SimulatedIdealAntenna,
    SimulatedLinearAntenna,
    SimulatedPatchAntenna,
    SimulatedUniformArray,
)
from .beamforming import DeviceFocus
from .drop import SimulatedDrop
from .evaluators import SidelobeEvaluator, SI, SSINR
from .scenario import SimulationScenario
from .simulation import Simulation
from .simulated_device import (
    SimulatedDeviceState,
    ProcessedSimulatedDeviceInput,
    SimulatedDevice,
    SimulatedDeviceOutput,
    SimulatedDeviceReceiveRealization,
    SimulatedDeviceTransmission,
    SimulatedDeviceReception,
    TriggerModel,
    TriggerRealization,
    RandomTrigger,
    StaticTrigger,
    SampleOffsetTrigger,
    TimeOffsetTrigger,
)
from .rf_chain import (
    AnalogDigitalConverter,
    Gain,
    GainControlType,
    AutomaticGainControl,
    QuantizerType,
    RfChain,
    PowerAmplifier,
    SalehPowerAmplifier,
    RappPowerAmplifier,
    ClippingPowerAmplifier,
    CustomPowerAmplifier,
    PhaseNoise,
    NoPhaseNoise,
    OscillatorPhaseNoise,
)
from .isolation import Isolation, SpecificIsolation, PerfectIsolation, SelectiveLeakage
from .noise import NoiseLevel, N0, SNR, NoiseModel, AWGN, AWGNRealization
from .coupling import Coupling, ImpedanceCoupling, PerfectCoupling
from .modem import (
    SCIdealChannelEstimation,
    SingleCarrierIdealChannelEstimation,
    OFDMIdealChannelEstimation,
    EBN0,
    ESN0,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "Trajectory",
    "LinearTrajectory",
    "StaticTrajectory",
    "SimulatedAntenna",
    "SimulatedAntennaPort",
    "SimulatedCustomArray",
    "SimulatedAntennaArray",
    "SimulatedDipole",
    "SimulatedIdealAntenna",
    "SimulatedLinearAntenna",
    "SimulatedPatchAntenna",
    "SimulatedUniformArray",
    "Simulation",
    "DeviceFocus",
    "SimulatedDrop",
    "SidelobeEvaluator",
    "SI",
    "SSINR",
    "SimulationScenario",
    "SimulatedDeviceState",
    "ProcessedSimulatedDeviceInput",
    "SimulatedDevice",
    "SimulatedDeviceOutput",
    "SimulatedDeviceReceiveRealization",
    "SimulatedDeviceTransmission",
    "SimulatedDeviceReception",
    "TriggerModel",
    "TriggerRealization",
    "RandomTrigger",
    "StaticTrigger",
    "SampleOffsetTrigger",
    "TimeOffsetTrigger",
    "AnalogDigitalConverter",
    "Gain",
    "GainControlType",
    "AutomaticGainControl",
    "QuantizerType",
    "RfChain",
    "PowerAmplifier",
    "SalehPowerAmplifier",
    "RappPowerAmplifier",
    "ClippingPowerAmplifier",
    "CustomPowerAmplifier",
    "PhaseNoise",
    "NoPhaseNoise",
    "OscillatorPhaseNoise",
    "AnalogDigitalConverter",
    "Gain",
    "GainControlType",
    "AutomaticGainControl",
    "QuantizerType",
    "isolation",
    "Isolation",
    "SpecificIsolation",
    "PerfectIsolation",
    "SelectiveLeakage",
    "NoiseLevel",
    "N0",
    "SNR",
    "NoiseModel",
    "AWGN",
    "AWGNRealization",
    "Coupling",
    "ImpedanceCoupling",
    "PerfectCoupling",
    "SCIdealChannelEstimation",
    "SingleCarrierIdealChannelEstimation",
    "OFDMIdealChannelEstimation",
    "EBN0",
    "ESN0",
]
