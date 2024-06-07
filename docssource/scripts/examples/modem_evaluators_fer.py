# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.modem import FrameErrorEvaluator, TransmittingModem, ReceivingModem, RootRaisedCosineWaveform
from hermespy.simulation import Simulation


# Create a new simulation featuring two devices
simulation = Simulation()
device_alpha = simulation.new_device()
device_beta = simulation.new_device()

# Create a transmitting and receiving modem for each device, respectively
modem_alpha = TransmittingModem(device=device_alpha)
modem_beta = ReceivingModem(device=device_beta)

# Configure the modem's waveform
waveform_configuration = {
    'symbol_rate': 1e8,
    'num_preamble_symbols': 10,
    'num_data_symbols': 100,
}
modem_alpha.waveform = RootRaisedCosineWaveform(**waveform_configuration)
modem_beta.waveform = RootRaisedCosineWaveform(**waveform_configuration)

simulation.add_evaluator(FrameErrorEvaluator(modem_alpha, modem_beta))
simulation.new_dimension('noise_level', dB(0, 2, 4, 8, 10, 12, 14, 16, 18, 20), device_beta)
simulation.num_samples = 1000
result = simulation.run()

result.plot()
plt.show()
