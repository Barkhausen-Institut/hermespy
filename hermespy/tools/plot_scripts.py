"""
This script collects the data from several results folders and plot in a single figure the BER/FER plots for several
simulations
"""
import os
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np

########################################################################################################################
# parameters
# choose the desired plots here

# common path where the results can be found
results_path = os.path.join(os.getcwd(), "../..", "results")

# list with the folders containing the simulation results
dirs = ["SISO", "MRC", "SFBC_2tx"]

# label for each simulation
labels = ["SISO", "MRC (1x2)", "SFBC (2x1)"]

# line styles
styles = ['-*', '-o', '-v', '-s', '-8', '-^', '-x', '-p', '-D', '-h']

# True if confidence interval bars should be displayed
plot_confidence_interval = True
########################################################################################################################

if len(dirs) != len(labels):
    raise ValueError("'labels' and 'dirs' must have the same length")

if len(styles) < len(dirs):
    styles = styles * int(np.ceil((len(dirs) / len(styles))))

plt.figure()
for results_dir, label, style in zip(dirs, labels, styles):
    file_path = os.path.join(results_path, results_dir, 'statistics.mat')
    res = sio.loadmat(file_path)

    snr = res['snr_vector'].flatten()
    ber = res['ber_mean'].flatten()

    if plot_confidence_interval:
        error = np.vstack((ber - res['ber_lower'].flatten(), res['ber_upper'].flatten() - ber))
        plt.errorbar(snr, ber, error, fmt=style, label=label)
    else:
        plt.plot(snr, ber, style, label=label)

plt.yscale("log", nonposy="mask")
plt.legend()

plt.xlabel("Eb/N0(dB)")
plt.ylabel("BER")

plt.show()
