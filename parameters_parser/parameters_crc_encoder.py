from parameters_parser.parameters_encoder import ParametersEncoder


class ParametersCrcEncoder(ParametersEncoder):
    def __init__(self, crc_bits: int = 0) -> None:
        self.crc_bits = crc_bits
        self.data_bits_k = 1
        self.encoded_bits_n = self.data_bits_k + self.crc_bits

    def check_params(self) -> None:
        """checks the validity of the parameters."""
        if self.crc_bits < 0:
            raise ValueError("Crc bits must be larger than 0.")
