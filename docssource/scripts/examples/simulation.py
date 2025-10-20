# -*- coding: utf-8 -*-

import numpy as np

from hermespy.core import dB, ReceivePowerEvaluator, Signal, SignalReceiver, SignalTransmitter
from hermespy.channel import IdealChannel
from hermespy.simulation import Simulation

# Alias the ideal channel as Channel
Channel = IdealChannel

# Initialize a new simulation
simulation = Simulation()

# Limit the number of actors, i.e. parallel physical layers, to 3
simulation.num_actors = 3

# Add two devices to the simulation
device_alpha = simulation.new_device()
device_beta = simulation.new_device()

# Configure the three unique channels
alpha_alpha_channel = Channel()
beta_beta_channel = Channel()
alpha_beta_channel = Channel()

simulation.set_channel(device_alpha, device_alpha, alpha_alpha_channel)
simulation.set_channel(device_beta, device_beta, beta_beta_channel)
simulation.set_channel(device_alpha, device_beta, alpha_beta_channel)

# Sweep over the SNR from 0 to 20 dB in steps of 10 dB
simulation.new_dimension("noise_level", dB(0, 10, 20), device_beta)

# Sweep over the carrier frequency from 1 GHz to 100 GHz in steps of 10 GHz
simulation.new_dimension("carrier_frequency", [1e9, 1e10, 1e11], device_alpha, device_beta)

# Make both devices transmit 100 samples at 100 MHz
ns, fs = 100, 1e8
transmitted_signal = Signal.Create(np.exp(2j * np.random.uniform(0, np.pi, (1, ns))), fs)
alpha_transmitter = SignalTransmitter(transmitted_signal)
beta_transmitter = SignalTransmitter(transmitted_signal)
alpha_receiver = SignalReceiver(ns, expected_power=1.0)
beta_receiver = SignalReceiver(ns, expected_power=1.0)

device_alpha.transmitters.add(alpha_transmitter)
device_beta.transmitters.add(beta_transmitter)
device_alpha.receivers.add(alpha_receiver)
device_beta.receivers.add(beta_receiver)

# Add an evaluator estimating the received power to the simulation
simulation.add_evaluator(ReceivePowerEvaluator(alpha_receiver))
simulation.add_evaluator(ReceivePowerEvaluator(beta_receiver))

# Run the simulation
result = simulation.run()

# Print the result and plot graphs
result.print()
result.plot()

# Save the result to a file
result.save_to_matlab("simulation.mat")
