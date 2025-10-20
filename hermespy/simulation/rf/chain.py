# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Generic, Iterable, SupportsIndex, SupportsInt
from typing_extensions import override

from hermespy.core import (
    DenseSignal,
    RandomNode,
    Serializable,
    SerializationProcess,
    DeserializationProcess,
)
from .block import (
    DSPInputBlock,
    DSPOutputBlock,
    RFBlock,
    RFBlockPort,
    RFBlockPortType,
    RFBlockRealization,
    RFBT,
)
from .signal import RFSignal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RFChainRealization(object):
    """Realization of a radio-frequency chain."""

    __bandwidth: float
    __oversampling_factor: int
    __sampling_rate: float
    __carrier_frequency: float
    __block_states: dict[RFBlockReference, RFBlockRealization]

    def __init__(
        self,
        bandwidth: float,
        oversampling_factor: int,
        carrier_frequency: float,
        block_states: dict[RFBlockReference, RFBlockRealization],
    ) -> None:
        """
        Args:
            bandwidth: Simulation bandwidth of the chain in Hz.
            oversampling_factor: Oversampling factor of the chain's modeling.
            carrier_frequency: Target carrier frequency of the modeled radio-frequency chain in Hz.
            block_states: Dictionary mapping block references to their states.
        """

        # Store attributes
        self.__bandwidth = bandwidth
        self.__oversampling_factor = oversampling_factor
        self.__sampling_rate = bandwidth * oversampling_factor
        self.__carrier_frequency = carrier_frequency
        self.__block_states = block_states

    @property
    def bandwidth(self) -> float:
        """Simulation bandwidth of the chain in Hz."""

        return self.__bandwidth

    @property
    def oversampling_factor(self) -> int:
        """Oversampling factor of the chain's modeling."""

        return self.__oversampling_factor

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the chain in Hz."""

        return self.__sampling_rate

    @property
    def carrier_frequency(self) -> float:
        """Target carrier frequency of the modeled radio-frequency chain in Hz."""

        return self.__carrier_frequency

    @property
    def block_states(self) -> dict[RFBlockReference, RFBlockRealization]:
        """List of realizations for each block in the radio-frequency chain."""

        return self.__block_states.copy()


class RFBlockReference(Generic[RFBT], Serializable):
    """Reference to a radio-frequency block within a radio-frequency chain."""

    __block: RFBT
    __incoming_connections: OrderedDict[RFBlockReference, tuple[list[int], list[int]]]
    __outgoing_connections: OrderedDict[RFBlockReference, tuple[list[int], list[int]]]

    def __init__(self, block: RFBT) -> None:
        """
        Args:
            block: The radio-frequency block to reference.
        """

        self.__block = block
        self.__outgoing_connections = OrderedDict()
        self.__incoming_connections = OrderedDict()

    @property
    def block(self) -> RFBT:
        """The referenced radio-frequency block."""

        return self.__block

    @property
    def outgoing_connections(self) -> OrderedDict[RFBlockReference, tuple[list[int], list[int]]]:
        """Dictionary of blocks connected to this block's output ports.

        The map's keys are references to connected blocks.
        The map's values are tuples containing:
        * A list of outgoing port indices from this block.
        * A list of incoming port indices to the connected block.
        """

        return self.__outgoing_connections.copy()

    @property
    def incoming_connections(self) -> OrderedDict[RFBlockReference, tuple[list[int], list[int]]]:
        """Dictionary of blocks connected to this block's input ports.

        The map's keys are references to connected blocks.
        The map's values are tuples containing:
        * A list of outgoing port indices from the connected block.
        * A list of incoming port indices to this block.
        """

        return self.__incoming_connections.copy()

    @property
    def num_connected_input_ports(self) -> int:
        """Number of connected input ports."""

        return sum(len(v[0]) for v in self.__incoming_connections.values())

    @property
    def num_connected_output_ports(self) -> int:
        """Number of connected output ports."""

        return sum(len(v[0]) for v in self.__outgoing_connections.values())

    @property
    def open_input_ports(self) -> list[int]:
        """List of open input port indices."""

        port_indices = list(range(self.block.num_input_ports))
        for _, connection_port_indices in self.__incoming_connections.values():
            for port_index in connection_port_indices:
                port_indices.remove(port_index)

        return port_indices

    def incoming_block_index(self, block: RFBlockReference) -> int:
        """Get the index of the incoming block in the list of incoming connections.

        Args:
            block: The radio-frequency block reference to find in the incoming connections.

        Returns: The index of the incoming block in the list of incoming connections.
        """

        # This might be a performance bottleneck and can be optimized if necessary
        return list(self.__incoming_connections.keys()).index(block)

    def connect_to(
        self, input: RFBlockReference, output_ports: int | list[int], input_ports: int | list[int]
    ) -> None:
        """Connect this block's output ports to another block's input ports.

        Args:
            input: The input block to connect to.
            output_ports: The output ports of this block to connect.
            input_ports: The input ports of the input block to connect.
        """

        _input_ports = [input_ports] if isinstance(input_ports, int) else input_ports
        _output_ports = [output_ports] if isinstance(output_ports, int) else output_ports

        # Ensure the port indices are valid
        if any(port < 0 or port >= self.block.num_output_ports for port in _output_ports):
            raise ValueError(f"Invalid output port index {_output_ports}")
        if input.block.num_input_ports > 0 and any(
            port < 0 or port >= input.block.num_input_ports for port in _input_ports
        ):
            raise ValueError(f"Invalid input port index {_input_ports}")

        self.__outgoing_connections[input] = (_output_ports, _input_ports)

    def connect_from(
        self, output: RFBlockReference, output_ports: int | list[int], input_ports: int | list[int]
    ) -> None:
        """Connect this block to another block's input ports.

        Args:
            output: The output block to connect to.
            output_ports: The output ports of this block to connect.
            input_ports: The input ports of the output block to connect.
        """

        _output_ports = [output_ports] if isinstance(output_ports, int) else output_ports
        _input_ports = [input_ports] if isinstance(input_ports, int) else input_ports

        # Ensure the port indices are valid
        if any(p < 0 or p >= output.block.num_output_ports for p in _output_ports):
            raise ValueError(f"Invalid output port index {_output_ports}")
        if self.block.num_input_ports > 0 and any(
            p < 0 or p >= self.block.num_input_ports for p in _input_ports
        ):
            raise ValueError(f"Invalid input port index {_input_ports}")

        self.__incoming_connections[output] = (_output_ports, _input_ports)

    def propagate(self, state: RFBlockRealization, input: RFSignal) -> RFSignal:
        """Propagate a signal through the referenced block.

        Args:
            state: The state of the referenced block.
            input: The input signal to propagate through the block.

        Returns: The output signal after propagation through the block.
        """

        return self.block.propagate(state, input)

    def port(self, name: str) -> RFBlockPortReference[RFBT]:
        """Get a reference to a port or group of ports by name.

        Args:
            name:
                Name of the port or group of ports to access.
                The name resolves to an attribute of the referenced of type :class:`RFBlockPort<hermespy.simulation.rf.block.RFBlockPort>`.

        Returns:
            A reference to the requested port or group of ports.

        Raises:
            AttributeError: If the requested port or group of ports does not exist.
        """

        if not hasattr(self.__block, name):
            raise AttributeError(f"Block {self.__block} has no port or group of ports named {name}")

        block_attribute = getattr(self.__block, name)
        if not isinstance(block_attribute, RFBlockPort):
            raise AttributeError(f"Attribute {name} of block {self.__block} is not a port or group")

        return RFBlockPortReference(self, block_attribute.port_indices, block_attribute.port_type)

    def __getattribute__(self, name: str) -> object:
        """Override attribute access to return the block's attributes directly.

        Args:
            name: The name of the attribute to access.

        Returns: The value of the attribute from the block if it exists, otherwise from the reference itself.
        """

        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if hasattr(self.__block, name):
            block_attribute = getattr(self.__block, name)
            if isinstance(block_attribute, RFBlockPort):
                return RFBlockPortReference(
                    self, block_attribute.port_indices, block_attribute.port_type
                )

        return object.__getattribute__(self, name)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__block, "block")

    @classmethod
    @override
    def Deserialize(
        cls: type[RFBlockReference], process: DeserializationProcess
    ) -> RFBlockReference:
        block = process.deserialize_object("block", RFBlock)
        return RFBlockReference(block)


class RFBlockPortReference(Generic[RFBT], RFBlockPort[RFBT]):
    """Reference to a port or a group of ports of a radio-frequency block."""

    __block_reference: RFBlockReference[RFBT]

    def __init__(
        self,
        block_reference: RFBlockReference[RFBT],
        port_indices: SupportsInt | Iterable[int],
        port_type: RFBlockPortType,
    ) -> None:
        """
        Args:
            block_reference: Reference to the radio-frequency block this port belongs to.
            port_indices: Integer index or sequence of indices of the represented port(s).
            port_type: Type of the port, either input or output.
        """

        # Initialize base class
        RFBlockPort.__init__(self, block_reference.block, port_indices, port_type)

        # Store the block reference
        self.__block_reference = block_reference

    @property
    def block_reference(self) -> RFBlockReference[RFBT]:
        """Reference to the radio-frequency block this port belongs to."""

        return self.__block_reference

    @override
    def __getitem__(self, index: slice | SupportsIndex) -> RFBlockPortReference[RFBT]:

        # Select the port index subset
        selected_port_indices = self.port_indices[index]

        # Return a new port instance representing the selected port(s)
        return RFBlockPortReference(self.block_reference, selected_port_indices, self.port_type)


class RFChain(RandomNode, Serializable):
    """Representation of a block-base radio-frequency front-end."""

    __block_references: list[RFBlockReference]
    __dsp_input_blocks: list[RFBlockReference]
    __dsp_output_blocks: list[RFBlockReference]
    __rf_input_blocks: list[RFBlockReference]
    __rf_output_blocks: list[RFBlockReference]
    __rf_origin_blocks: list[RFBlockReference]

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:
            seed: Seed with which to initialize the radio-frequency chain's random state.
        """

        # Initialize base classes
        Serializable.__init__(self)
        RandomNode.__init__(self, seed=seed)

        # Initialize class attributes
        self.__block_references = []
        self.__dsp_input_blocks = []
        self.__dsp_output_blocks = []
        self.__rf_input_blocks = []
        self.__rf_output_blocks = []
        self.__rf_origin_blocks = []

    @property
    def num_digital_input_ports(self) -> int:
        """Number of digital ports feeding into the radio-frequency chain."""

        return sum(b.block.num_input_ports for b in self.__dsp_input_blocks)

    @property
    def num_digital_output_ports(self) -> int:
        """Number of digital ports receiving output from the radio-frequency chain."""

        return sum(b.block.num_output_ports for b in self.__dsp_output_blocks)

    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFChainRealization:
        """Realize the radio-frequency chain.

        Args:
            sampling_rate: Simulation bandwidth of the chain in Hz.
            oversampling_factor: Oversampling factor of the chain's modeling.
            carrier_frequency: Target carrier frequency of the modeled radio-frequency chain in Hz.

        Returns: An instance of :class:`RFChainRealization` representing the realized chain.
        """

        # Generate block states
        block_realizations: dict[RFBlockReference, RFBlockRealization] = {
            r: r.block.realize(bandwidth, oversampling_factor, carrier_frequency)
            for r in self.__block_references
        }

        return RFChainRealization(
            bandwidth, oversampling_factor, carrier_frequency, block_realizations
        )

    def add_block(self, block: RFBT) -> RFBlockReference[RFBT]:
        """Add a new radio-frequency block to the chain.

        The same block configuration can be added multiple times,
        each time creating a new :class:`BlockReference<hermespy.simulation.rf.chain.RFBlockReference>` instance.

        Args:
            block: The radio-frequency block to add to the chain.

        Returns: A reference to the added block.
        """

        block_reference = RFBlockReference(block)
        self.__block_references.append(block_reference)

        # Add blocks without RF input ports to the origin blocks list
        if block.num_input_ports < 1:
            self.__rf_origin_blocks.append(block_reference)

        # Add DSP blocks to the respective lists
        if block.num_input_ports > 0:
            if isinstance(block, DSPInputBlock):
                self.__dsp_input_blocks.append(block_reference)
            else:
                self.__rf_input_blocks.append(block_reference)
        if block.num_output_ports > 0:
            if isinstance(block, DSPOutputBlock):
                self.__dsp_output_blocks.append(block_reference)
            else:
                self.__rf_output_blocks.append(block_reference)

        # Return the final block reference
        return block_reference

    def add_blocks(self, block: RFBT, count: int) -> list[RFBlockReference[RFBT]]:
        """Add multiple identical blocks the radio-frequency chain.

        Args:
            block: Radio frequency block to add to the chain.
            count: Number of times to add the block.

        Returns: List of references to the added blocks.
        """

        return [self.add_block(block) for _ in range(count)]

    def new_block(self, block: type[RFBT], **kwargs) -> RFBlockReference[RFBT]:
        """Initialize a new radio-frequency block instance and add it to the chain.

        Args:
            block: Type of the radio-frequency block to create.
            kwargs: Additional keyword arguments to pass to the block's constructor.

        Returns:
            A reference to the newly initialized block instance.
        """

        # Create a new block instance
        new_block = block(**kwargs)

        # Add the new block to the chain
        return self.add_block(new_block)

    def new_blocks(self, block: type[RFBT], count: int, **kwargs) -> list[RFBlockReference[RFBT]]:
        """Initialize multiple identical radio-frequency block instances and add them to the chain.

        Args:
            block: Type of the radio-frequency block to create.
            count: Number of blocks to create.
            kwargs: Additional keyword arguments to pass to the block's constructor.

        Returns: List of references to the newly created blocks.
        """

        # Create a new block instance
        new_block = block(**kwargs)

        # Add the new blocks to the chain
        return self.add_blocks(new_block, count)

    def connect(self, port_a: RFBlockPortReference, port_b: RFBlockPortReference) -> None:
        """Connect an output port of one block to an input port of another block.

        Args:
            port_a: First port to connect.
            port_b: Second port to connect.

        Raises:
            ValueError: If two ports of the same type are connected.
        """

        # Ensure a connection from outputs to inputs
        if port_a.port_type == port_b.port_type:
            raise ValueError(
                f"Cannot connect two ports of the same type, both are of type {port_a.port_type}"
            )

        # Ensure both port references expose the same number of ports
        if port_a.num_ports != port_b.num_ports:
            raise ValueError(
                f"Cannot connect ports with different number of ports: {port_a.num_ports} and {port_b.num_ports}"
            )

        input_port: RFBlockPortReference
        output_port: RFBlockPortReference
        if port_a.port_type == RFBlockPortType.IN:
            input_port = port_a
            output_port = port_b
        else:
            input_port = port_b
            output_port = port_a

        output_block = output_port.block_reference
        input_block = input_port.block_reference

        # Register connection from output block ports to input block ports
        output_block.connect_to(
            input_port.block_reference, output_port.port_indices, input_port.port_indices
        )

        # Register connection from input block ports to output block ports
        input_block.connect_from(
            output_port.block_reference, output_port.port_indices, input_port.port_indices
        )

        # Remove the involved block references from the input/output lists
        # if all their ports are connected
        if output_block.num_connected_output_ports >= output_block.block.num_output_ports:
            if output_block in self.__rf_output_blocks:
                self.__rf_output_blocks.remove(output_block)
        if input_block.num_connected_input_ports >= input_block.block.num_input_ports:
            if input_block in self.__rf_input_blocks:
                self.__rf_input_blocks.remove(input_block)

    def transmit(
        self, dsp_output: DenseSignal | RFSignal, realization: RFChainRealization
    ) -> RFSignal:
        """Transmit a signal through the radio-frequency chain.

        Args:
            dsp_output: Samples generated by the transmitting DSP layer.
            realization: Realization of the radio-frequency chain.

        Returns:
            The output signal after propagation through the radio-frequency chain.
        """

        _dsp_output: RFSignal = (
            dsp_output if isinstance(dsp_output, RFSignal) else RFSignal.FromDense(dsp_output)
        ).resample(realization.sampling_rate)
        num_samples = _dsp_output.num_samples

        origin_inputs: MutableMapping[RFBlockReference, DenseSignal] = {}
        p = 0
        for block in self.__dsp_input_blocks:
            num_input_ports = block.block.num_input_ports
            origin_inputs[block] = _dsp_output[p : p + block.block.num_input_ports, :]
            p += num_input_ports

        # Propagate back from the DSP output through the chain
        num_output_ports = sum(b.block.num_output_ports for b in self.__rf_output_blocks)
        output = RFSignal(num_output_ports, num_samples, realization.sampling_rate)
        k = 0
        block_outputs: MutableMapping[RFBlockReference, DenseSignal] = {}
        for block in self.__rf_output_blocks:
            block_output = self.__generate_output(
                realization, num_samples, block, origin_inputs, block_outputs
            )
            output[k : k + block_output.num_streams, :] = block_output
            k += block_output.num_streams

        return output

    def receive(
        self, rf_input: RFSignal | DenseSignal, realization: RFChainRealization
    ) -> DenseSignal:
        """Receive a signal through the radio-frequency chain.

        Args:
            rf_input: Samples received by the radio-frequency chain.
            realization: Realization of the radio-frequency chain.

        Returns:
            The output signal after propagation through the radio-frequency chain.
        """

        _rf_input = rf_input if isinstance(rf_input, RFSignal) else RFSignal.FromDense(rf_input)
        # _rf_input = _rf_input.resample(realization.sampling_rate)  TODO for RFSignal
        num_samples = _rf_input.num_samples

        origin_inputs: MutableMapping[RFBlockReference, DenseSignal] = {}
        p = 0
        for block_ref in self.__rf_input_blocks:
            num_block_input_ports = len(block_ref.open_input_ports)
            origin_inputs[block_ref] = _rf_input[p : p + num_block_input_ports, :]
            p += num_block_input_ports

        # Propagate forward from the RF input through the chain
        num_input_ports = sum(b.block.num_input_ports for b in self.__dsp_output_blocks)
        output = RFSignal(num_input_ports, num_samples, realization.sampling_rate)
        k = 0
        block_outputs: MutableMapping[RFBlockReference, DenseSignal] = {}
        for block in self.__dsp_output_blocks:
            block_output = self.__generate_output(
                realization, num_samples, block, origin_inputs, block_outputs
            )
            output[k : k + block_output.num_streams, : block_output.num_samples] = block_output
            k += block_output.num_streams

        return output

    def __generate_output(
        self,
        chain_realization: RFChainRealization,
        num_samples: int,
        block: RFBlockReference,
        origin_inputs: MutableMapping[RFBlockReference, DenseSignal],
        block_outputs: MutableMapping[RFBlockReference, DenseSignal],
    ) -> DenseSignal:
        """Generate the output signal for a specific block.

        Args:
            block: The block reference to fetch the output from.
            block_outputs: Dictionary of block references and their corresponding output signals.

        Returns: The output signal of the specified block.
        """

        # If the output is already available, simply return it
        if block in block_outputs:
            return block_outputs[block]

        # Generate the block's input signals
        connected_block_ouputs = [
            self.__generate_output(chain_realization, num_samples, o, origin_inputs, block_outputs)
            for o in block.incoming_connections.keys()
        ]

        # Combine all relevant inputs into a single DenseSignal
        block_input = RFSignal(
            block.block.num_input_ports, num_samples, chain_realization.sampling_rate
        )
        for impinging_signal, (output_ports, input_ports) in zip(
            connected_block_ouputs, block.incoming_connections.values()
        ):
            _num_samples = min(num_samples, impinging_signal.num_samples)
            block_input[input_ports, :_num_samples] = impinging_signal[output_ports, :_num_samples]

        # Add origin inputs if available
        if block in origin_inputs:
            origin_input = origin_inputs[block]
            block_input[block.open_input_ports, : origin_input.num_samples] = origin_input

        # Propagate the input signals through the block
        block_state = chain_realization.block_states[block]
        block_output = block.propagate(block_state, block_input)

        # Store the output signal in the block outputs dictionary
        block_outputs[block] = block_output
        return block_output

    @override
    def serialize(self, process: SerializationProcess) -> None:
        # Serialize the invidivdual block references
        process.serialize_object_sequence(self.__block_references, "block_references")
        process.serialize_object_sequence(self.__dsp_input_blocks, "dsp_input_blocks")
        process.serialize_object_sequence(self.__dsp_output_blocks, "dsp_output_blocks")
        process.serialize_object_sequence(self.__rf_input_blocks, "rf_input_blocks")
        process.serialize_object_sequence(self.__rf_output_blocks, "rf_output_blocks")
        process.serialize_object_sequence(self.__rf_origin_blocks, "rf_origin_blocks")

        # Serialize random seed if available
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> RFChain:
        chain_instance = RFChain(seed=process.deserialize_integer("seed", None))

        # Deserialize the individual block references
        chain_instance.__block_references = list(
            process.deserialize_object_sequence("block_references", RFBlockReference)
        )
        chain_instance.__dsp_input_blocks = list(
            process.deserialize_object_sequence("dsp_input_blocks", RFBlockReference)
        )
        chain_instance.__dsp_output_blocks = list(
            process.deserialize_object_sequence("dsp_output_blocks", RFBlockReference)
        )
        chain_instance.__rf_input_blocks = list(
            process.deserialize_object_sequence("rf_input_blocks", RFBlockReference)
        )
        chain_instance.__rf_output_blocks = list(
            process.deserialize_object_sequence("rf_output_blocks", RFBlockReference)
        )
        chain_instance.__rf_origin_blocks = list(
            process.deserialize_object_sequence("rf_origin_blocks", RFBlockReference)
        )

        return chain_instance
