# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from hermespy.radar import FMCW
from hermespy.simulation import SimulatedDeviceState

waveform = FMCW(num_chirps=1)

waveform.ping(SimulatedDeviceState.Basic(bandwidth=1e8)).plot(space='time', title='FMCW bandwidth=1e8')
waveform.ping(SimulatedDeviceState.Basic(bandwidth=3e9)).plot(space='time', title='FMCW bandwidth=3e9')

plt.show()
