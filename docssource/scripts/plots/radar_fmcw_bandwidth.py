# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from hermespy.radar import FMCW

waveform = FMCW(num_chirps=1)

waveform.bandwidth = 1e8
waveform.ping().plot(space='time', title='FMCW bandwidth=1e8')

waveform.bandwidth = 3e9
waveform.ping().plot(space='time', title='FMCW bandwidth=3e9')

plt.show()
