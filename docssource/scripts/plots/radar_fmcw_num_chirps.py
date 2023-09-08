# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from hermespy.radar import FMCW

waveform = FMCW()

waveform.num_chirps = 1
waveform.ping().plot(space='time', title='FMCW num_chirps=1')

waveform.num_chirps = 3
waveform.ping().plot(space='time', title='FMCW num_chirps=3')

plt.show()
