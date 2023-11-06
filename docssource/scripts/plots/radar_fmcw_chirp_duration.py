# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from hermespy.radar import FMCW

waveform = FMCW(num_chirps=2)

waveform.chirp_duration = 1.5e-6
waveform.ping().plot(space='time', title='FMCW chirp_duration=1.5e-6, pulse_rep_interval=1.5e-6')

waveform.chirp_duration = 1e-6
waveform.ping().plot(space='time', title='FMCW chirp_duration=1e-6, pulse_rep_interval=1.5e-6')

plt.show()
