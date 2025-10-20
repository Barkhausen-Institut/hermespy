# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.modem import ConstellationEVM, TransmittingModem, ReceivingModem, RootRaisedCosineWaveform
from hermespy.simulation import Simulation


# Create a new simulation featuring two devices
simulation = Simulation()
device_alpha = simulation.new_device(bandwidth=1e8, oversampling_factor=8)
device_beta = simulation.new_device(bandwidth=1e8, oversampling_factor=8)

# Create a transmitting and receiving modem for each device, respectively
modem_alpha = TransmittingModem()
device_alpha.transmitters.add(modem_alpha)
modem_beta = ReceivingModem()
device_beta.receivers.add(modem_beta)

# Configure the modem's waveform
waveform_configuration = {
    'num_preamble_symbols': 10,
    'num_data_symbols': 100,
}
modem_alpha.waveform = RootRaisedCosineWaveform(**waveform_configuration)
modem_beta.waveform = RootRaisedCosineWaveform(**waveform_configuration)

simulation.add_evaluator(ConstellationEVM(modem_alpha, modem_beta))
simulation.new_dimension('noise_level', dB(0, 2, 4, 8, 10, 12, 14, 16, 18, 20), device_beta)
simulation.num_samples = 1000
result = simulation.run()

result.plot()
plt.show()
