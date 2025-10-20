# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from hermespy.radar import FMCW
from hermespy.simulation import SimulatedDeviceState

waveform = FMCW()

waveform.num_chirps = 1
waveform.ping(SimulatedDeviceState.Basic(bandwidth=1e9)).plot(space='time', title='FMCW num_chirps=1')

waveform.num_chirps = 3
waveform.ping(SimulatedDeviceState.Basic(bandwidth=1e9)).plot(space='time', title='FMCW num_chirps=3')

plt.show()
