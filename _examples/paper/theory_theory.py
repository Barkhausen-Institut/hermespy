# -*- coding: utf-8 -*-

from os import mkdir
from os.path import join

import numpy as np
from numpy import exp
from scipy.io import savemat
from scipy import stats
from scipy.special import comb

from hermespy.tools import db2lin
from hermespy.core import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def sc_awgn(snrs: np.ndarray,
            modulation_order: int) -> np.ndarray:

    bits_per_symbol = np.log2(modulation_order)

    # BPSK/QPSK
    if modulation_order in [2, 4]:
        ber = stats.norm.sf(np.sqrt(2 * snrs))

    # M-PSK
    elif modulation_order == 8:
        ber = (2 * stats.norm.sf(np.sqrt(2 * bits_per_symbol * snrs) *
                                    np.sin(np.pi / modulation_order)) / bits_per_symbol)

    # M-QAM
    elif modulation_order in [16, 64, 256]:
        ser = 4 * (np.sqrt(modulation_order) - 1) / np.sqrt(modulation_order) * \
                stats.norm.sf(np.sqrt(3 * bits_per_symbol / (modulation_order - 1) * snrs))
        ber = ser / bits_per_symbol

    else:
        return None

    return ber

def sc_rayleigh(snrs: np.ndarray,
                modulation_order: int) -> np.ndarray:

    # BPSK
    if modulation_order == 2:

        alpha = 1
        beta = 2.

    # M-PSK
    elif modulation_order in [4, 8]:

        alpha = 2
        beta = 2 * np.sin(np.pi / modulation_order) ** 2

    # M-QAM
    elif modulation_order in [16, 64, 256]:

        alpha = 4 * (np.sqrt(modulation_order) -1) / np.sqrt(modulation_order)
        beta = 3 / (modulation_order - 1)

    else:
        return None

    ser = alpha / 2 * (1 - np.sqrt(beta * snrs / 2 / (1 + beta * snrs / 2)))
    return ser / np.log2(modulation_order)

def fsk_awgn(snrs: np.ndarray,
             mod_order):

    # For modulation orders greater than 64 the implemented method produces numerical errors
    if mod_order > 64:
        return None

    n_bits = np.log2(mod_order)
    ebn0_linear = snrs

    # calculate BER according do Proakis, Salehi, Digital
    # Communications, 5th edition, Section 4.5, Equations 44 and 47
    ser = np.zeros(len(ebn0_linear))  # symbol error rate
    for n in range(2, mod_order+1):
        ser += (-1)**n / n * exp(- (n - 1) * n_bits / n * ebn0_linear) * comb(mod_order - 1, n - 1)

    # Bit error rate
    ber = 2 ** (n_bits - 1) / (2 ** n_bits - 1) * ser
    return ber


modulation_orders = [2, 4, 16, 64]
snrs = np.array([db2lin(x) for x in np.arange(-10, 20, .5)])


directory_prefix = Executable.default_results_dir()


awgn_theory = np.empty((len(modulation_orders), len(snrs)))
rayleigh_theory = np.empty((len(modulation_orders), len(snrs)))
fsk_theory = np.empty((len(modulation_orders), len(snrs)))
for m, modulation_order in enumerate(modulation_orders):
    
    awgn_theory[m, :] = sc_awgn(snrs, modulation_order)
    rayleigh_theory[m, :] = sc_rayleigh(snrs, modulation_order)
    fsk_theory[m, :] = fsk_awgn(snrs, modulation_order)


awgn_dir = join(directory_prefix, 'validation', 'awgn', 'sc_theory')
rayleigh_dir = join(directory_prefix, 'validation', 'rayleigh', 'sc_theory')
fsk_dir = join(directory_prefix, 'validation', 'awgn', 'fsk_theory')

try:
    mkdir(awgn_dir)
except FileExistsError:
    ...
    
try:
    mkdir(rayleigh_dir)
except FileExistsError:
    ...
    
try:
    mkdir(fsk_dir)
except FileExistsError:
    ...

savemat(join(awgn_dir, 'results.mat'), {'theory': awgn_theory})
savemat(join(rayleigh_dir, 'results.mat'), {'theory': rayleigh_theory})
savemat(join(fsk_dir, 'results.mat'), {'theory': fsk_theory})
