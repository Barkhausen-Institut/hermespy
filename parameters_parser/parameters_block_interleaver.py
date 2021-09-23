from parameters_parser.parameters_encoder import ParametersEncoder


class ParametersBlockInterleaver(ParametersEncoder):
    def __init__(self, M: int, N: int) -> None:
        self.M = M
        self.N = N
        self.encoded_bits_n = M * N
        self.data_bits_k = self.encoded_bits_n

    def check_params(self) -> None:
        """checks the validity of the parameters."""
        if self.M < 1 or self.N < 1:
            raise ValueError("M or N must be larger than 0.")
