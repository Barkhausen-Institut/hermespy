# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Transformation
from hermespy.channel import IndoorFactory, FactoryType
from hermespy.modem import TransmittingModem, ReceivingModem, OFDMWaveform,  SymbolSection, GridResource, GridElement
from hermespy.simulation import Simulation

simulation = Simulation()
device_alpha = simulation.new_device(carrier_frequency=1e8, pose=Transformation.From_Translation(np.zeros(3)))
device_beta = simulation.new_device(carrier_frequency=1e8, pose=Transformation.From_Translation(np.array([100, 0, 0])))

tx_modem = TransmittingModem(device=device_alpha)
rx_modem = ReceivingModem(device=device_beta)

ofdm_resources = [GridResource(elements=[GridElement('DATA', 1024)])]
ofdm_structure = [SymbolSection(10, [0])]
waveform = OFDMWaveform(grid_resources=ofdm_resources, grid_structure=ofdm_structure)

tx_modem.waveform = waveform
rx_modem.waveform = waveform

channel = IndoorFactory(15000, 60000, FactoryType.HH)

transmission = device_alpha.transmit()
propagation = channel.propagate(transmission, device_alpha, device_beta)
reception = device_beta.receive(propagation)

transmission.mixed_signal.plot(title='Transmitted Signal')


plt.show()
