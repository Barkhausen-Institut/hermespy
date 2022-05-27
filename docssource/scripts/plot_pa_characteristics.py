# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.constants import pi

from hermespy.core import Executable
from hermespy.simulation import PowerAmplifier, SalehPowerAmplifier, RappPowerAmplifier, \
    ClippingPowerAmplifier
    
__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


with Executable.style_context():
    
    figure, amplitude_axes = plt.subplots()
    figure.suptitle("Power Amplifier Model Characteristics")

    amplitude_axes.set_xlabel("Input Amplitude")
    amplitude_axes.set_ylabel("Output Amplitude")

    phase_axes = amplitude_axes.twinx()
    phase_axes.set_ylabel("Output Phase")
    phase_axes.set_ylim([-pi, pi])

    saturation_amplitude = 1.0
    amplifiers = [
        PowerAmplifier(saturation_amplitude=saturation_amplitude),
        RappPowerAmplifier(saturation_amplitude=saturation_amplitude, smoothness_factor=0.5),
        ClippingPowerAmplifier(saturation_amplitude=saturation_amplitude),
        SalehPowerAmplifier(saturation_amplitude=saturation_amplitude, amplitude_alpha=1.9638, amplitude_beta=0.9945, phase_alpha=2.5293, phase_beta=2.8168),
    ]

    for amplifier in amplifiers:
        amplifier.plot(axes=(amplitude_axes, phase_axes))
        
    amplitude_axes.legend(['Linear', 'Rapp', 'Clipping', 'Saleh'])
    phase_axes.set_visible(False)

    plt.show()