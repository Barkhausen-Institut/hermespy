from typing import Dict, Any

import numpy as np


def compare_theory_simulation_errors(
        error_mean: np.array, error_upper: np.array, error_lower: np.array) -> bool:
    results_equal = True
    for error_mean_rx, error_upper_rx, error_lower_rx in zip(
            error_mean, error_upper, error_lower):
        results_equal = results_equal & np.all(
            error_mean_rx <= error_upper_rx) & np.all(
            error_mean_rx >= error_lower_rx)
    return results_equal


def compare_mat_files(
        simulation_mat: Dict[str, Any], expected_results_mat: Dict[str, Any], no_tx_modems: int = 1) -> bool:
    results_equal = True

    if not nan_arrays_equal(
            simulation_mat["ber_mean"], expected_results_mat["ber_mean"]):
        results_equal = False
    elif not nan_arrays_equal(simulation_mat["fer_mean"], expected_results_mat["fer_mean"]):
        results_equal = False
    elif not nan_arrays_equal(simulation_mat["ber_lower"], expected_results_mat["ber_lower"]):
        results_equal = False
    elif not nan_arrays_equal(simulation_mat["fer_lower"], expected_results_mat["fer_lower"]):
        results_equal = False
    elif not nan_arrays_equal(simulation_mat["ber_upper"], expected_results_mat["ber_upper"]):
        results_equal = False
    elif not nan_arrays_equal(simulation_mat["fer_upper"], expected_results_mat["fer_upper"]):
        results_equal = False

    return results_equal


def nan_arrays_equal(a: np.array, b: np.array) -> bool:
    """Compares input arrays by replacing `np.nan` with `-1`."""

    a[np.isnan(a)] = -1
    b[np.isnan(b)] = -1

    equal = np.allclose(np.around(a, decimals=3), np.around(b, decimals=3))
    return equal
