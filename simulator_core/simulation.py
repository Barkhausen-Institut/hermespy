# -*- coding: utf-8 -*-
"""HermesPy simulation configuration."""

from __future__ import annotations
from typing import List, Type
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from .executable import Executable
from .drop import Drop

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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
                 plot_drop: bool = True) -> None:
        """Object initialization.

        Args:
            plot_drop (bool, optional): Pause to plot each drop during execution.
        """

        Executable.__init__(self, plot_drop)

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

                plt.show()

    @classmethod
    def to_yaml(cls: Type[Simulation],
                representer: SafeRepresenter,
                node: Simulation) -> MappingNode:
        """Serialize an `Simulation` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (WaveformGenerator):
                The `Simulation` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "plot_drop": node.plot_drop,
        }

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
        return cls(**state)
