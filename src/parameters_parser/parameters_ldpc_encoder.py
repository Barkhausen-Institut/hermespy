import os
import configparser
from fractions import Fraction

from parameters_parser.parameters_encoder import ParametersEncoder


class ParametersLdpcEncoder(ParametersEncoder):
    SUPPORTED_CODE_RATES = [Fraction(1, 3),
                            Fraction(1, 2),
                            Fraction(2, 3),
                            Fraction(3, 4),
                            Fraction(4, 5),
                            Fraction(5, 6)]

    SUPPORTED_BLOCK_SIZES = [256, 512, 1024, 2048, 4096, 8192]

    def read_params(self, config: configparser.SectionProxy) -> None:
        super().read_params(config)
        self.block_size = config.getint('block_size', fallback=1)  # type: ignore
        self.no_iterations = config.getint('no_iterations', fallback=20)  # type: ignore
        self.code_rate_fraction = Fraction(self.code_rate).limit_denominator(6)
        self.custom_ldpc_codes = config.get('custom_ldpc_codes', fallback="")
        self.use_binding = config.getboolean('use_binding')  # type: ignore

        self.check_params()

    def check_params(self) -> None:
        super().check_params()
        if self.custom_ldpc_codes != "":
            if not os.path.exists(self.custom_ldpc_codes):
                raise ValueError(f"Path {self.custom_ldpc_codes} does not exist.")
        else:
            if self.code_rate_fraction not in self.SUPPORTED_CODE_RATES:
                raise ValueError(f"Supported code rates are: {self.SUPPORTED_CODE_RATES}, you provide {self.code_rate_fraction}")

            if self.block_size not in self.SUPPORTED_BLOCK_SIZES:
                raise ValueError(f"Supported block sizes are: {self.SUPPORTED_BLOCK_SIZES}, you provide {self.block_size}")

        if self.block_size < 0:
            raise ValueError("Block size must be positive.")

        if self.no_iterations < 0:
            raise ValueError("Number of iterations must be positive.")
