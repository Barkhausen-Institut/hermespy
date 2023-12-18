import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Signal

sampling_rate = 100
timestamps = np.arange(100) / sampling_rate
signal = Signal(np.exp(-.25j * np.pi * timestamps * sampling_rate), sampling_rate)

signal.plot()
plt.show()