# -*- coding: utf-8 -*-
# This simulation models a joint communication and sensing scenario.
#
# We assume a base station communicating with a terminal using a single-carrier
# waveform. Simulataneously, the base-station infers spatial information from
# its backscattered communication signal.
#
# The simulation evaluates both the bit error rate of the downling commuication
# between base station and terminal as well as the probability of detection of an
# object within the base-stations vicinity.

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import dB
from hermespy.simulation import (
    Simulation,
    N0,
    SpecificIsolation,
    StaticTrajectory,
)
from hermespy.channel import (
    MultiTargetRadarChannel,
    FixedCrossSection,
    TDL,
    TDLType,
)
from hermespy.jcas import MatchedFilterJcas
from hermespy.modem import (
    RectangularWaveform,
    ReceivingModem,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierCorrelationSynchronization,
    SingleCarrierZeroForcingChannelEqualization,
    BitErrorEvaluator,
)
from hermespy.radar import ThresholdDetector, ReceiverOperatingCharacteristic

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Initialize a simulation considering a base station and a terminal
cf = 1e9
simulation = Simulation()
bandwidth = 1e8
oversampling_factor = 4
base_station = simulation.new_device(carrier_frequency=cf, bandwidth=bandwidth, oversampling_factor=oversampling_factor)
terminal = simulation.new_device(carrier_frequency=cf, bandwidth=bandwidth, oversampling_factor=oversampling_factor)

# Assume a 100 dB transmit-receive isolation at the base station
base_station.isolation = SpecificIsolation(dB(100))

# Place terminal and base station 20 m apart
base_station.trajectory = StaticTrajectory.From_Translation(np.zeros(3))
terminal.trajectory = StaticTrajectory.From_Translation(np.array([20, 0, 0]))

# Configure a rectangular single-carrier waveform
waveform = RectangularWaveform(
    modulation_order=16,
    num_preamble_symbols=16,
    num_data_symbols=128,
    pilot_rate=10,
    guard_interval=1e-6,
)

# Configure the base-station to transmit the rectangular waveform
# and use environmental reflections to estimate a radar range-power profile
jcas_dsp = MatchedFilterJcas(
    max_range=30,
    waveform=deepcopy(waveform),
)
jcas_dsp.detector = ThresholdDetector(.95, peak_detection=False)
base_station.add_dsp(jcas_dsp)

# Configure the terminal to receive the rectangular waveform
terminal_dsp = ReceivingModem(
    waveform=deepcopy(waveform),
)
terminal_dsp.waveform.synchronization = SingleCarrierCorrelationSynchronization()
terminal_dsp.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
terminal_dsp.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
terminal.add_dsp(terminal_dsp)

# Configure a spatial radar channel for the base station's tx-rx path
# Assign the terminal an RCS of 10 m²
radar_channel = MultiTargetRadarChannel()
terminal_target = radar_channel.make_target(terminal, FixedCrossSection(10))
simulation.set_channel(base_station, base_station, radar_channel)

# Configuer a 5G TDL channel for the communication link between base station and terminal
communication_channel = TDL(TDLType.A)
simulation.set_channel(base_station, terminal, communication_channel)

# Configure the simulation to evaluate the bit error rate of the communication
simulation.add_evaluator(BitErrorEvaluator(jcas_dsp, terminal_dsp))

# Configure the simulation to estimate the base station's ROC
simulation.add_evaluator(ReceiverOperatingCharacteristic(jcas_dsp, base_station, base_station, radar_channel))

# Sweep over the device's noise power
base_station.noise_level = N0(dB(-100))
terminal.noise_level = N0(dB(-100))
simulation.new_dimension("noise_level", dB(range(-100, 10, 10)), base_station, terminal)

# Run the simulation and visualize the results
simulation.num_samples = 1000
result = simulation.run()
result.plot()
result.print()
plt.show()
