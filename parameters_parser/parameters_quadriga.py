import os
import configparser


class ParametersQuadriga:
    def __init__(self, dir_settings: str) -> None:
        self.matlab_or_octave = ""
        self.scenario_label = ""
        self.path_quadriga_src = ""
        self.antenna_kind = ""
        self.multipath_model = "QUADRIGA"
        self._dir_settings = os.path.dirname(dir_settings)

    def read_params(self, section: configparser.SectionProxy) -> None:
        config = configparser.ConfigParser()
        config.read(os.path.join(self._dir_settings, 'settings_quadriga.ini'))

        self.quadriga_executor = config["General"].get("quadriga_executor")
        self.path_quadriga_src = os.path.join(
            os.path.dirname(__file__), '..', '3rdparty', 'quadriga_srcs')

        self.scenario_label = config["Scenario"].get("scenario_label")
        self.antenna_kind = config["Scenario"].get("antenna_kind")

        self.check_params()

    def check_params(self) -> None:
        if (self.quadriga_executor not in ["matlab", "octave"] or
                not os.path.exists(self.path_quadriga_src)):
            raise ValueError("Wrong Parameter Values")
