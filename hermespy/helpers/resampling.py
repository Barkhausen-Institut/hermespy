# -*- coding: utf-8 -*-
"""HermesPy resampling routines."""

from math import ceil

import numpy as np

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def delay_resampling_matrix(sampling_rate: float,
                            num_samples: int,
                            delay: float) -> np.ndarray:
    """Generate an interpolation-matrix for resampling a signal at a specific delay.

    Args:

        sampling_rate (float):
            Rate in Hz at which the signal to be transformed is sampled.

        num_samples (int):
            Number of samples provided.

        delay (float):
            Delay in seconds, by which the sampled signal should be shifted.

    Returns:
        np.ndarray:
            A MxN linear resampling transformation matrix, where M is the number of input samples
            and N is the number of resampled output samples.
            Due to the delay, M might be bigger (or smaller for negative delays) than N, so that
            the transformation matrix is not necessarily square.
    """

    delay_samples_overhead = int(ceil(abs(delay) * sampling_rate)) * np.sign(delay)
    input_timestamps = np.arange(num_samples)
    output_timestamps = np.arange(num_samples + delay_samples_overhead) - delay * sampling_rate

    interpolation_filter = np.sinc(np.subtract.outer(output_timestamps, input_timestamps))
    return interpolation_filter