# -*- coding: utf-8 -*-
"""
=============
Hardware Loop
=============
"""

from __future__ import annotations
from typing import Type

from ruamel.yaml import SafeRepresenter, SafeConstructor, Node

from hermespy.core import Executable
from hermespy.core.factory import Serializable
from hermespy.core.scenario import Scenario
from .physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HardwareLoop(Executable, Scenario[PhysicalDevice]):
    """Hermespy hardware loop configuration.
    """

    yaml_tag = u'Loop'

    def __init__(self) -> None:

        Executable.__init__(self)
        Scenario.__init__(self)

    def run(self) -> None:

        # Trigger devices
        for device in self.devices:
            device.trigger()

    def from_yaml(cls: Type[Serializable], constructor: SafeConstructor, node: Node) -> Serializable:
        pass

    def to_yaml(cls: Type[Serializable], representer: SafeRepresenter, node: Serializable) -> Node:
        pass