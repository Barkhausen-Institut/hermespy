# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Transformation
from hermespy.channel import IndoorFactoryLineOfSight
from hermespy.modem import TransmittingModem, ReceivingModem, OFDMWaveform,  FrameSymbolSection, FrameResource, FrameElement
from hermespy.simulation import Simulation

simulation = Simulation()
device_alpha = simulation.new_device(pose=Transformation.From_Translation(np.zeros(3)))
device_beta = simulation.new_device(pose=Transformation.From_Translation(np.array([100, 0, 0])))

tx_modem = TransmittingModem(device=device_alpha)
rx_modem = ReceivingModem(device=device_beta)

ofdm_resources = [FrameResource(elements=[FrameElement('DATA', 1024)])]
ofdm_structure = [FrameSymbolSection(10, [0])]
waveform = OFDMWaveform(resources=ofdm_resources, structure=ofdm_structure)

tx_modem.waveform = waveform
rx_modem.waveform = waveform

channel = IndoorFactoryLineOfSight(15000, 60000, device_alpha, device_beta)

transmission = device_alpha.transmit()
propagation = channel.propagate(transmission)
reception = device_beta.receive(propagation)

transmission.mixed_signal.plot(title='Transmitted Signal')


plt.show()
