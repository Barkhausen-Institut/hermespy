# -*- coding: utf-8 -*-
"""HermesPy resampling routines."""

from math import ceil

import numpy as np

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.4"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def delay_resampling_matrix(sampling_rate: float,
                            num_samples_in: int,
                            delay: float,
                            num_samples_out: int = -1) -> np.ndarray:
    """Generate an interpolation-matrix for resampling a signal at a specific delay.

    Args:

        sampling_rate (float):
            Rate in Hz at which the signal to be transformed is sampled.

        num_samples_in (int):
            Number of samples provided.

        delay (float):
            Delay in seconds, by which the sampled signal should be shifted.

        num_samples_out(int, optional):
            Number of output samples.

    Returns:
        np.ndarray:
            A MxN linear resampling transformation matrix, where M is the number of input samples
            and N is the number of resampled output samples.
            Due to the delay, M might be bigger (or smaller for negative delays) than N, so that
            the transformation matrix is not necessarily square.
    """

    input_timestamps = np.arange(num_samples_in)

    if num_samples_out < 0:
        delay_samples_overhead = int(ceil(abs(delay) * sampling_rate)) * np.sign(delay)
        output_timestamps = np.arange(num_samples_in + delay_samples_overhead) - delay * sampling_rate

    else:
        output_timestamps = np.arange(num_samples_out) - delay * sampling_rate

    interpolation_filter = np.sinc(np.subtract.outer(output_timestamps, input_timestamps))
    return interpolation_filter
