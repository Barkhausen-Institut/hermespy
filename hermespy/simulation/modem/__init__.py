# -*- coding: utf-8 -*-

from .channel_estimation import SingleCarrierIdealChannelEstimation, OFDMIdealChannelEstimation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Class name aliasing
SCIdealChannelEstimation = SingleCarrierIdealChannelEstimation

__all__ = [
    "OFDMIdealChannelEstimation",
    "SCIdealChannelEstimation",
    "SingleCarrierIdealChannelEstimation",
]
