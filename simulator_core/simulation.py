# -*- coding: utf-8 -*-
"""HermesPy simulation configuration."""

from __future__ import annotations
from typing import List, Type, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode
from enum import Enum

from .executable import Executable
from .drop import Drop
from channel import QuadrigaInterface

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SNRType(Enum):
    """Supported signal-to-noise ratio types."""

    EBN0 = 0
    ESN0 = 1
    CUSTOM = 2


class SimulationDrop(Drop):
    """Data generated within a single simulation drop."""

    def __init__(self,
                 transmitted_bits: List[np.ndarray],
                 transmitted_signals: List[np.ndarray],
                 received_signals: List[np.ndarray],
                 received_bits: List[np.ndarray]) -> None:
        """Object initialization.

        Args:
            transmitted_bits (List[np.ndarray]): Bits fed into the transmitting modems.
            transmitted_signals (List[np.ndarray]): Modulated signals emitted by transmitting modems.
            received_signals (List[np.ndarray]): Modulated signals impinging onto receiving modems.
            received_bits (List[np.ndarray]): Bits output by receiving modems.
        """

        Drop.__init__(self, transmitted_bits, transmitted_signals, received_signals, received_bits)


class Simulation(Executable):
    """HermesPy simulation configuration."""

    yaml_tag = u'Simulation'

    def __init__(self,
                 plot_drop: bool = False,
                 calc_transmit_spectrum: bool = False,
                 calc_receive_spectrum: bool = False,
                 calc_transmit_stft: bool = False,
                 calc_receive_stft: bool = False,
                 snr_type: Union[str, SNRType] = SNRType.EBN0,
                 noise_loop: Union[List[float], np.ndarray] = np.array([0.0])) -> None:
        """Simulation object initialization.

        Args:
            plot_drop (bool): Plot each drop during execution of scenarios.
            calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
            calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
            calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
            calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
            snr_type (Union[str, SNRType]): The signal to noise ratio metric to be used.
            noise_loop (Union[List[float], np.ndarray]): Loop over different noise levels.
        """

        Executable.__init__(self, plot_drop, calc_transmit_spectrum, calc_receive_spectrum,
                            calc_transmit_stft, calc_receive_stft)

        self.snr_type = snr_type
        self.noise_loop = noise_loop

    def run(self) -> None:
        """Run the full simulation configuration."""

        drops: List[SimulationDrop] = []

        # Iterate over scenarios
        for scenario in self.scenarios:

            # Generate data bits to be transmitted
            data_bits = scenario.generate_data_bits()

            # Generate radio-frequency band signal emitted from each transmitter
            transmitted_signals = scenario.transmit(data_bits=data_bits)

            # Simulate propagation over channel models
            propagated_signals = scenario.propagate(transmitted_signals)

            # Receive and demodulate signal
            received_bits = scenario.receive(propagated_signals)

            # Save generated signals
            drop = SimulationDrop(data_bits, transmitted_signals, propagated_signals, received_bits)
            drops.append(drop)

            # Visualize plot if requested
            if self.plot_drop:

                drop.plot_transmitted_bits()
                drop.plot_transmitted_signals()
                drop.plot_received_signals()
                drop.plot_received_bits()
                drop.plot_bit_errors()

                plt.show()

    @property
    def snr_type(self) -> SNRType:
        """Type of signal-to-noise ratio.

        Returns:
            SNRType: The SNR type.
        """

        return self.__snr_type

    @snr_type.setter
    def snr_type(self, snr_type: Union[str, SNRType]) -> None:
        """Modify the type of signal-to-noise ratio.

        Args:
            snr_type (Union[str, SNRType]):
                The new type of signal to noise ratio, string or enum representation.
        """

        if isinstance(snr_type, str):
            snr_type = SNRType[snr_type]

        self.__snr_type = snr_type

    @property
    def noise_loop(self) -> np.ndarray:
        """Access the configured signal to noise ratios over which the simulation iterates.

        Returns:
            np.ndarray: The signal to noise ratios.
        """

        return self.__noise_loop

    @noise_loop.setter
    def noise_loop(self, loop: Union[List[float], np.ndarray]) -> None:
        """Modify the configured signal to noise ratios over which the simulation iterates.
        
        Args:
            loop (Union[List[float], np.ndarray]): The new noise loop.

        Raises:
            ValueError: If `loop` does not represent a vector with at least one entry.
        """

        # Convert lists to arrays
        if isinstance(loop, List):
            loop = np.array(loop, dtype=float)

        if loop.ndim != 1:
            raise ValueError("The noise loop must be a vector")

        if len(loop) < 1:
            raise ValueError("The noise loop must contain at least one SNR entry")

        self.__noise_loop = loop

    @classmethod
    def to_yaml(cls: Type[Simulation],
                representer: SafeRepresenter,
                node: Simulation) -> MappingNode:
        """Serialize an `Simulation` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Simulation):
                The `Simulation` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "plot_drop": node.plot_drop,
            "snr_type": node.snr_type.value,
            "noise_loop": node.noise_loop.tolist()
        }

        # If a global Quadriga interface exists,
        # add its configuration to the simulation section
        if QuadrigaInterface.GlobalInstanceExists():
            state[QuadrigaInterface.yaml_tag] = QuadrigaInterface.GlobalInstance()

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[Simulation],
                  constructor: SafeConstructor,
                  node: MappingNode) -> Simulation:
        """Recall a new `Simulation` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Simulation` serialization.

        Returns:
            WaveformGenerator:
                Newly created `Simulation` instance.
        """

        state = constructor.construct_mapping(node)

        # Launch a global quadriga instance
        quadriga_interface: Optional[QuadrigaInterface] = state.pop(QuadrigaInterface.yaml_tag, None)
        if quadriga_interface is not None:
            QuadrigaInterface.SetGlobalInstance(quadriga_interface)

        return cls(**state)
