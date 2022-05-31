# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Executable
from hermespy.simulation.analog_digital_converter import AnalogDigitalConverter, QuantizerType, GainControlType
    
__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


with Executable.style_context():
    
    figure, fig_axes = plt.subplots(1, 2)
    figure.suptitle("ADC Quantization Model")

    quant = AnalogDigitalConverter(num_quantization_bits=3,
                            gain_control=GainControlType.NONE,
                            max_amplitude=1.,
                            quantizer_type=QuantizerType.MID_RISER)

    input_sig = np.arange(-1.2, 1.2, .001) + 1j * np.arange(1.2, -1.2, -0.001)

    quant.plot_quantizer(input_samples=input_sig, fig_axes=fig_axes[0],
                         label='Mid-Riser')

    quant.quantizer_type = QuantizerType.MID_TREAD

    quant.plot_quantizer(input_samples=input_sig, fig_axes=fig_axes[1],
                         label='Mid-Tread')
    plt.show()
