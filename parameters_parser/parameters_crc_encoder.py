from parameters_parser.parameters_encoder import ParametersEncoder


class ParametersCrcEncoder(ParametersEncoder):
    def __init__(self, crc_bits: int = 0,
                       k_following_encoder: int = 1) -> None:
        self.crc_bits = crc_bits
        self.encoded_bits_n = k_following_encoder
        self.data_bits_k = self.encoded_bits_n - self.crc_bits

    def check_params(self) -> None:
        """checks the validity of the parameters."""
        if self.crc_bits < 0:
            raise ValueError("Crc bits must be larger than 0.")

        if self.encoded_bits_n < 0:
            raise ValueError("Crc is bigger than k of following encoder.")
