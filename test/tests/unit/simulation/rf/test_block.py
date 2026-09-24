# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase
from unittest.mock import Mock
from typing_extensions import override

from hermespy.core import SerializationProcess, DeserializationProcess
from hermespy.simulation.rf.block import (
    RFBlock,
    RFBlockRealization,
    RFBlockPort,
    RFBlockPortType,
    ActiveRFBlock,
    PassiveRFBlock,
)
from hermespy.simulation import (
    NoiseLevel,
    NoiseModel,
    RFSignal,
    AWGN,
    N0,
    DCPowerModel,
    NoDCPowerModel,
    ConstantDCPowerModel,
    PowerAmplifier,
    Source,
    ADC,
    DAC,
    RampGenerator,
    Mixer,
    IdealMixer,
    Shift,
    X4,
    Split,
    Sum,
    HPF,
)
from ...core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockRFBlock(RFBlock):
    """Mock RF block for testing."""

    __num_input_ports: int
    __num_output_ports: int
    __i: RFBlockPort
    __o: RFBlockPort

    def __init__(
        self,
        num_input_ports: int = 1,
        num_output_ports: int = 1,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Initialize port attributes
        self.__num_input_ports = num_input_ports
        self.__num_output_ports = num_output_ports
        self.__i = RFBlockPort(self, range(num_input_ports), RFBlockPortType.IN)
        self.__o = RFBlockPort(self, range(num_output_ports), RFBlockPortType.OUT)

    @property
    def i(self) -> RFBlockPort:
        return self.__i

    @property
    def o(self) -> RFBlockPort:
        return self.__o

    @property
    @override
    def num_input_ports(self) -> int:
        return self.__num_input_ports

    @property
    @override
    def num_output_ports(self) -> int:
        return self.__num_output_ports

    @override
    def realize(self, bandwidth: float, oversampling_factor: int) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        return 2 * input

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_input_ports, "num_input_ports")
        process.serialize_integer(self.num_output_ports, "num_output_ports")
        process.serialize_object(self.noise_model, "noise_model")
        process.serialize_object(self.noise_level, "noise_level")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> "MockRFBlock":
        num_input_ports = process.deserialize_integer("num_input_ports")
        num_output_ports = process.deserialize_integer("num_output_ports")
        noise_model = process.deserialize_object("noise_model", NoiseModel)
        noise_level = process.deserialize_object("noise_level", NoiseLevel)
        seed = process.deserialize_integer("seed", None)
        return MockRFBlock(
            num_input_ports=num_input_ports,
            num_output_ports=num_output_ports,
            noise_model=noise_model,
            noise_level=noise_level,
            seed=seed,
        )


class TestRFBlock(TestCase):
    """Test the base class of all RF blocks."""

    def setUp(self) -> None:

        self.noise_model = AWGN(42)
        self.noise_level = N0(0.123)

        self.block = MockRFBlock(
            num_input_ports=2,
            num_output_ports=3,
            noise_level=self.noise_level,
            noise_model=self.noise_model,
            seed=42,
        )

    def test_init(self) -> None:
        """Test initialization of RF blocks"""

        self.assertEqual(self.block.num_output_ports, 3)
        self.assertEqual(self.block.noise_model, self.noise_model)
        self.assertEqual(self.block.noise_level, self.noise_level)
        self.assertEqual(self.block.seed, 42)

    def test_propagate_validation(self) -> None:
        """Propagate should raise a value error on invalid input streams"""

        mock_signal = Mock(spec=RFSignal)
        mock_signal.num_streams = 123

        with self.assertRaises(ValueError):
            self.block.propagate(self.block.realize(123, 4), mock_signal)

    def test_serialization(self) -> None:
        """Test serialization of RF blocks"""

        test_roundtrip_serialization(self, self.block)


class MockActiveRFBlock(ActiveRFBlock):
    """Mock active RF block for testing."""

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        return input

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.dc_power_model, "dc_power_model")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> MockActiveRFBlock:
        return cls(dc_power_model=process.deserialize_object("dc_power_model", DCPowerModel))


class MockPassiveRFBlock(PassiveRFBlock):
    """Mock passive RF block for testing."""

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        return input

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> "MockPassiveRFBlock":
        return cls()


class TestActiveRFBlock(TestCase):
    """Test the base class of all power-consuming RF blocks."""

    def setUp(self) -> None:
        self.block = MockActiveRFBlock()

    def test_default_dc_power_model(self) -> None:
        """Blocks without a configured model should not consume any power"""

        self.assertIsInstance(self.block.dc_power_model, NoDCPowerModel)

    def test_dc_power_model_setget(self) -> None:
        """Power model property getter should return setter argument"""

        expected_model = ConstantDCPowerModel(2.5)
        self.block.dc_power_model = expected_model
        self.assertIs(expected_model, self.block.dc_power_model)

    def test_dc_power_model_none(self) -> None:
        """Setting None should fall back to the placeholder model"""

        self.block.dc_power_model = ConstantDCPowerModel(2.5)
        self.block.dc_power_model = None
        self.assertIsInstance(self.block.dc_power_model, NoDCPowerModel)

    def test_serialization(self) -> None:
        """Test serialization of active RF blocks"""

        test_roundtrip_serialization(
            self, MockActiveRFBlock(dc_power_model=ConstantDCPowerModel(2.5))
        )


class TestPassiveRFBlock(TestCase):
    """Test the base class of all supply-independent RF blocks."""

    def setUp(self) -> None:
        self.block = MockPassiveRFBlock()

    def test_no_dc_power_model(self) -> None:
        """Passive blocks should not expose a power consumption model"""

        self.assertFalse(hasattr(self.block, "dc_power_model"))

class TestBlockClassification(TestCase):
    """Test the active / passive classification of the shipped RF blocks."""

    def test_active_blocks(self) -> None:
        """Blocks drawing direct current power should derive from ActiveRFBlock"""

        for block in (
            PowerAmplifier,
            Source,
            ADC,
            DAC,
            RampGenerator,

        ):
            self.assertTrue(issubclass(block, ActiveRFBlock), f"{block.__name__} should be active")

    def test_passive_blocks(self) -> None:
        """Blocks operating without a power supply should derive from PassiveRFBlock"""

        for block in (
            Split,
            Sum, 
            HPF,
            Mixer,
            IdealMixer,
            Shift,
            X4,
        ):
                self.assertTrue(issubclass(block, PassiveRFBlock), f"{block.__name__} should be passive")