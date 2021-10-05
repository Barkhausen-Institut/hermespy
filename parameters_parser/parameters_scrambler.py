import numpy as np
import configparser

from .parameters_encoder import ParametersEncoder


class ParametersScrambler(ParametersEncoder):

    def __init__(self) -> None:
        """creates a parsing object, that will manage the Encoder parameters."""

        ParametersEncoder.__init__(self)

        self.seed = np.array([0] * 7)

    def read_params(self, section: configparser.SectionProxy) -> None:
        """reads the channel parameters contained in the section 'section' of a given configuration file."""

        seed = section.get("seed", fallback="0,0,0,0,0,0,0")
        self.seed = np.fromstring(seed, dtype=float, sep=',')

    def check_params(self) -> None:
        """checks the validity of the parameters."""

        if self.seed.shape[0] != 7:
            raise ValueError("The seed must contain exactly 7 bit")

        for bit in self.seed:
            if bit != 0 and bit != 1:
                raise ValueError("Only bits (i.e. 0 or 1) represent valid seed fields")

