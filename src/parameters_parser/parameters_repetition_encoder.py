import configparser

from parameters_parser.parameters_encoder import ParametersEncoder


class ParametersRepetitionEncoder(ParametersEncoder):
    def read_params(self, config: configparser.SectionProxy) -> None:
        self.check_params()

    def check_params(self) -> None:
        super().check_params()
