# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from hermespy.simulation.rf_chain.power_amplifier import PowerAmplifier, SalehPowerAmplifier, RappPowerAmplifier, \
    ClippingPowerAmplifier

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


saturation_amplitude = 1.0
samples = np.arange(0, 2, 0.01)

power_amplifier = PowerAmplifier(saturation_amplitude=saturation_amplitude)
clipping_power_amplifier = ClippingPowerAmplifier(saturation_amplitude=saturation_amplitude)
rapp_power_amplifier = RappPowerAmplifier(saturation_amplitude=saturation_amplitude, smoothness_factor=2.)
saleh_power_amplifier = SalehPowerAmplifier(saturation_amplitude=saturation_amplitude, amplitude_alpha=1.9638, amplitude_beta=0.9945, phase_alpha=2.5293, phase_beta=2.8168)

amplitudes = np.empty((4, len(samples)), dtype=float)
amplitudes[0, :] = abs(power_amplifier.model(samples))
amplitudes[1, :] = abs(clipping_power_amplifier.model(samples))
amplitudes[2, :] = abs(rapp_power_amplifier.model(samples))
amplitudes[3, :] = abs(saleh_power_amplifier.model(samples))

savemat('D:\\hermes_paper\\pa_characteristics\\results.mat', {
    'x': samples,
    'y': amplitudes,
})

plt.plot(samples, amplitudes.T)
plt.show()