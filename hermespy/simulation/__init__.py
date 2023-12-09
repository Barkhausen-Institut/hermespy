# -*- coding: utf-8 -*-

from .antennas import (
    SimulatedAntenna,
    SimulatedAntennaPort,
    SimulatedCustomArray,
    SimulatedAntennas,
    SimulatedDipole,
    SimulatedIdealAntenna,
    SimulatedLinearAntenna,
    SimulatedPatchAntenna,
    SimulatedUniformArray,
)
from .simulation import Simulation, SimulationScenario
from .simulated_device import (
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
from .noise import Noise, AWGN
from .coupling import Coupling, ImpedanceCoupling, PerfectCoupling
from .modem import (
    SCIdealChannelEstimation,
    SingleCarrierIdealChannelEstimation,
    OFDMIdealChannelEstimation,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "SimulatedAntenna",
    "SimulatedAntennaPort",
    "SimulatedCustomArray",
    "SimulatedAntennas",
    "SimulatedDipole",
    "SimulatedIdealAntenna",
    "SimulatedLinearAntenna",
    "SimulatedPatchAntenna",
    "SimulatedUniformArray",
    "Simulation",
    "SimulationScenario",
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
    "Noise",
    "AWGN",
    "Coupling",
    "ImpedanceCoupling",
    "PerfectCoupling",
    "SCIdealChannelEstimation",
    "SingleCarrierIdealChannelEstimation",
    "OFDMIdealChannelEstimation",
]
