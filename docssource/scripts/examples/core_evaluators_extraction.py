# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB, SignalExtractor
from hermespy.modem import TransmittingModem, ReceivingModem, RRCWaveform
from hermespy.simulation import Simulation


# Create a new simulation featuring two devices
simulation = Simulation(num_samples=100)
device_alpha = simulation.new_device()
device_beta = simulation.new_device()

# Create a transmitting and receiving modem for each device, respectively
modem_alpha = TransmittingModem(waveform=RRCWaveform())
device_alpha.transmitters.add(modem_alpha)
modem_beta = ReceivingModem(waveform=RRCWaveform())
device_beta.receivers.add(modem_beta)

# Configure the simulation to extract generated signals
simulation.add_evaluator(SignalExtractor(modem_alpha))
simulation.add_evaluator(SignalExtractor(modem_beta))
result = simulation.run()
