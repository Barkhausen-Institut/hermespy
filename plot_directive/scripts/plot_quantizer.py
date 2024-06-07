# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Executable, Signal
from hermespy.simulation import AnalogDigitalConverter, QuantizerType

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


with Executable.style_context():

    figure, fig_axes = plt.subplots(1, 2)
    figure.suptitle("ADC Quantization Model")

    quant = AnalogDigitalConverter(
                            num_quantization_bits=3,
                            quantizer_type=QuantizerType.MID_RISER)

    input_sig = np.arange(-1.2, 1.2, .001) + 1j * np.arange(1.2, -1.2, -0.001)

    quant.plot_quantizer(input_samples=input_sig, fig_axes=fig_axes[0],
                         label='Mid-Riser')

    quant.quantizer_type = QuantizerType.MID_TREAD

    quant.plot_quantizer(input_samples=input_sig, fig_axes=fig_axes[1],
                         label='Mid-Tread')
    plt.show()
