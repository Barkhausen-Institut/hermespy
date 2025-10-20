import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import IdealChannel
from hermespy.simulation import SimulatedDevice
from hermespy.modem import SimplexLink, BitErrorEvaluator, OFDMWaveform, GridResource, GridElement, ElementType, SymbolSection, GuardSection, PrefixType

# model an LTE OFDM link with 20 MHz bandwidth
# 3 data time slots + 1 empty slot are transmitted
num_subcarriers = 2048
bandwidth = 20e6
oversampling_factor = 2
dc_suppression = True
modulation_order = 16

resource_block_0 = [GridElement(ElementType.REFERENCE), GridElement(ElementType.DATA, 5),
                    GridElement(ElementType.REFERENCE), GridElement(ElementType.DATA, 5)]
ofdm_symbol_0 = GridResource(prefix_ratio=160/2048, repetitions=100, elements=resource_block_0, prefix_type=PrefixType.CYCLIC)

ofdm_symbol_1 = GridResource(prefix_ratio=144/2048, repetitions=1, elements=[GridElement(ElementType.DATA, 1200)],
                             prefix_type=PrefixType.CYCLIC)

resource_block_2 = [GridElement(ElementType.DATA, 3),
                    GridElement(ElementType.REFERENCE), GridElement(ElementType.DATA, 5),
                    GridElement(ElementType.REFERENCE), GridElement(ElementType.DATA, 2)]
ofdm_symbol_2 = GridResource(prefix_ratio=144/2048, repetitions=100, elements=resource_block_2, prefix_type=PrefixType.CYCLIC)

resources = [ofdm_symbol_0, ofdm_symbol_1, ofdm_symbol_2]

time_slot = SymbolSection(pattern=[0, 1, 1, 1, 2, 1, 1])
structure = [time_slot, time_slot, time_slot, GuardSection(duration=0.5e-3)]

# Create two simulated devices acting as source and sink
tx_device = SimulatedDevice(bandwidth=bandwidth, oversampling_factor=oversampling_factor)
rx_device = SimulatedDevice(bandwidth=bandwidth, oversampling_factor=oversampling_factor)

# Set up a unidirectional link between both simulated devices
link = SimplexLink()
link.waveform = OFDMWaveform(
    modulation_order=modulation_order, num_subcarriers=num_subcarriers,
    dc_suppression=dc_suppression, grid_resources=resources, grid_structure=structure)
tx_device.transmitters.add(link)
rx_device.receivers.add(link)

# Plot the time-frequency grid of the OFDM waveform
link.waveform.plot_grid()

# Evaluate bit errors during transmission and visualize the received symbol constellation
evaluator = BitErrorEvaluator(link, link)

# Simulate a channel between the two devices
channel = IdealChannel()

# Simulate the signal transmission over the channel
transmission = tx_device.transmit()
propagation = channel.propagate(transmission, tx_device, rx_device)
reception = rx_device.receive(propagation)

# Visualize communication performance
evaluator.evaluate().visualize()
reception.operator_receptions[0].equalized_symbols.plot_constellation()
reception.baseband_signal.plot()
plt.show()
