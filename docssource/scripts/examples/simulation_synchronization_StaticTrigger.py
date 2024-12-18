# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt

from hermespy.modem import SimplexLink, RootRaisedCosineWaveform, BitErrorEvaluator, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization
from hermespy.simulation import Simulation, StaticTrigger

# Create a new simulation featuring two sets of two linked, synchronized devices
simulation = Simulation()

# Create devices
device_A_Tx = simulation.new_device(carrier_frequency=3.7e9)
device_A_Rx = simulation.new_device(carrier_frequency=3.7e9)
device_B_Tx = simulation.new_device(carrier_frequency=3.9e9)
device_B_Rx = simulation.new_device(carrier_frequency=3.9e9)

# Specify a root-raised cosine single carrier waveform
waveform = RootRaisedCosineWaveform(
    symbol_rate=400e6,
    oversampling_factor=4,
    roll_off=.9,
    num_preamble_symbols=0,
    num_data_symbols=1024,
    pilot_rate=7,
)
waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Link devices
link_A = SimplexLink(waveform=deepcopy(waveform))
link_B = SimplexLink(waveform=deepcopy(waveform))
device_A_Tx.transmitters.add(link_A)
device_A_Rx.receivers.add(link_A)
device_B_Tx.transmitters.add(link_B)
device_B_Rx.receivers.add(link_B)

# Specify trigger models
trigger_model_A = StaticTrigger()
device_A_Tx.trigger_model = trigger_model_A
device_A_Rx.trigger_model = trigger_model_A

trigger_model_B = StaticTrigger()
device_B_Tx.trigger_model = trigger_model_B
device_B_Rx.trigger_model = trigger_model_B

# Configure BER evaluators for each link
ber_A = BitErrorEvaluator(link_A, link_A)
ber_B = BitErrorEvaluator(link_B, link_B)

# Generate drop
_ = simulation.scenario.drop()

# Visualize BERs that should spike at the frame overlaps
ber_A.evaluate().visualize(title='Link A Bit Error Rate')
ber_B.evaluate().visualize(title='Link B Bit Error Rate')
plt.show()
