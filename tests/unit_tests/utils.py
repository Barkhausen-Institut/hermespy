# -*- coding: utf-8 -*-

from typing import List
import re

import numpy as np

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def flatten_blocks(bits: List[np.array]) -> np.array:
    flattened_blocks = np.array([])
    for block in bits:
        flattened_blocks = np.append(flattened_blocks, block)
    return flattened_blocks


def assert_frame_equality(data_bits: List[np.array], encoded_bits: List[np.array]) -> None:
    for data_block, encoded_block in zip(data_bits, encoded_bits):
        np.testing.assert_array_equal(data_block, encoded_block)


def yaml_str_contains_element(yaml_str: str, key: float, value: float) -> bool:
    regex = re.compile(f"^\s*{key}: {value}\s*$", re.MULTILINE)

    return re.search(regex, yaml_str) is not None
