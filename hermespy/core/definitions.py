# -*- coding: utf-8 -*-

from .factory import SerializableEnum

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ConsoleMode(SerializableEnum):
    """Printing behaviour of the simulation during runtime"""

    INTERACTIVE = 0
    """Interactive refreshing of the shell information"""

    LINEAR = 1
    """Linear appending of the shell information"""

    SILENT = 2
    """No prints exept errors"""


class Verbosity(SerializableEnum):
    """Information output behaviour configuration of an executable"""

    ALL = 0
    """Print absolutely everything"""

    INFO = 1
    """Print general information"""

    WARNING = 2
    """Print only warnings and errors"""

    ERROR = 3
    """Print only errors"""

    NONE = 4
    """Print absolutely nothing"""


class FloatingError(RuntimeError):
    """Exception raised if an operation fails due to a currently being considered floating."""

    ...  # pragma: no cover


class InterpolationMode(SerializableEnum):
    """Interpolation behaviour for sampling and resampling routines.

    Considering a complex time series

    .. math::

       \\mathbf{s} = \\left[s_{0}, s_{1},\\,\\dotsc, \\, s_{M-1} \\right]^{\\mathsf{T}} \\in \\mathbb{C}^{M} \\quad \\text{with} \\quad s_{m} = s(\\frac{m}{f_{\\mathrm{s}}})

    sampled at rate :math:`f_{\\mathrm{s}}`, so that each sample
    represents a discrete sample of a time-continuous underlying function :math:`s(t)`.

    Given only the time-discrete sample vector :math:`\\mathbf{s}`,
    resampling refers to

    .. math::

       \\hat{s}(\\tau) = \\mathscr{F} \\left\\lbrace \\mathbf{s}, \\tau \\right\\rbrace

    estimating a sample of the original time-continuous function at time :math:`\\tau` given only the discrete-time sample vector :math:`\\mathbf{s}`.
    """

    NEAREST = 0
    """Interpolate to the nearest sampling instance.

    .. math::

       \\hat{s}(\\tau) = s_{\\lfloor \\tau f_{\\mathrm{s}} \\rfloor}

    Very fast, but not very accurate.
    """

    SINC = 1
    """Interpolate using sinc kernels.

    Also known as the Whittaker-Kotel'nikov-Shannon interpolation formula :footcite:p:`2002:meijering`.

    .. math::

       \\hat{s}(\\tau) = \\sum_{m=0}^{M-1} s_{m} \\operatorname{sinc} \\left( \\tau f_{\\mathrm{s}} - m \\right)

    Perfect for bandlimited signals, not very fast.
    """
