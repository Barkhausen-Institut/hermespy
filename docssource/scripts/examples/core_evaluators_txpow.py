# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB, TransmitPowerEvaluator
from hermespy.modem import TransmittingModem, ReceivingModem, RRCWaveform
from hermespy.simulation import Simulation


# Create a new simulation featuring two devices
simulation = Simulation()
device_alpha = simulation.new_device()
device_beta = simulation.new_device()

# Create a transmitting and receiving modem for each device, respectively
modem_alpha = TransmittingModem(waveform=RRCWaveform())
device_alpha.transmitters.add(modem_alpha)
modem_beta = ReceivingModem(waveform=RRCWaveform())
device_beta.receivers.add(modem_beta)

# Configure the simulation
simulation.add_evaluator(TransmitPowerEvaluator(modem_alpha))
simulation.new_dimension('noise_level', dB(0, 2, 4, 8, 10, 12, 14, 16, 18, 20), device_beta)
simulation.num_samples = 1000
result = simulation.run()

result.plot()
plt.show()
