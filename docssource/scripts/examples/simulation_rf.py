# -*- coding: utf-8 -*-

from hermespy.simulation import (
    RFChain,
    SimulatedDevice,
    ADC as DSPInputBlock,
    DAC as DSPOutputBlock,
    Shift as RFBlock_A,
    Mixer as RFBlock_B,
    Source as RFBlock_C,
    MixerType as Type
)


# Initialize the RF chain
chain = RFChain()

# Generate unconnected block references
input_ref = chain.add_block(DSPInputBlock())
output_ref = chain.add_block(DSPOutputBlock())
tx_feed_ref = chain.add_block(RFBlock_A())
tx_combiner_ref = chain.add_block(RFBlock_B(Type.UP))
rx_combiner_ref = chain.add_block(RFBlock_B(Type.DOWN))
source_ref = chain.add_block(RFBlock_C())

# Connect blocks
chain.connect(output_ref.port('o'), tx_feed_ref.port('i'))
chain.connect(tx_feed_ref.port('o'), tx_combiner_ref.port('i'))
chain.connect(source_ref.port('o'), tx_combiner_ref.port('lo'))
chain.connect(source_ref.port('o'), rx_combiner_ref.port('lo'))
chain.connect(rx_combiner_ref.port('o'), input_ref.port('i'))

# Initialize a device and assigne the RF chain to it
device = SimulatedDevice(carrier_frequency=1e9)
device.rf = chain

# Plot the RF chain's parameters
print(f'Number of DSP input ports: {chain.num_digital_input_ports}')
print(f'Number of DSP output ports: {chain.num_digital_output_ports}')
