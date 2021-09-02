from abc import ABC, abstractmethod


class ParametersWaveformGenerator(ABC):
    """This abstract class implements the parser of the waveform generator parameters."""

    @abstractmethod
    def __init__(self) -> None:
        """creates a parsing object, that will manage the waveform generator parameters."""
        # Modulation parameters
        self.sampling_rate = 0.

    @abstractmethod
    def read_params(self, file_name: str) -> None:
        """Reads the waveform generator parameters contained in the configuration file 'file_name'."""
        pass

    @abstractmethod
    def _check_params(self) -> None:
        """checks the validity of the parameters."""
        pass
