# -*- coding: utf-8 -*-

import numpy as np
from scipy.fft import ifft, ifftshift

from hermespy.simulation import Simulation, SelectiveLeakage

# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Specify the device's isolation model with a high-pass characteristic
leakage_frequency_response = np.zeros((1, 1, 101))
leakage_frequency_response[0, 0, :50] = np.linspace(1, 0, 50, endpoint=False)
leakage_frequency_response[0, 0, 50:] = np.linspace(0, 1, 51, endpoint=True)
leakage_impulse_response = ifft(ifftshift(leakage_frequency_response))
device.isolation = SelectiveLeakage(leakage_impulse_response)
