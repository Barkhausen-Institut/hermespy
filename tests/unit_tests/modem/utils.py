from typing import List

import numpy as np


def flatten_blocks(bits: List[np.array]) -> np.array:
    flattened_blocks = np.array([])
    for block in bits:
        flattened_blocks = np.append(
            flattened_blocks,
            block
        )
    return flattened_blocks
