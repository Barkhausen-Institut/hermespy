import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import IdealChannel
from hermespy.simulation import SimulatedDevice
from hermespy.modem import SimplexLink, BitErrorEvaluator, OFDMWaveform, FrameResource, FrameElement, ElementType, FrameSymbolSection, FrameGuardSection, PrefixType

# model an LTE OFDM link with 20 MHz bandwidth
# 3 data time slots + 1 empty slot are transmitted
subcarrier_spacing = 15e3
num_subcarriers = 2048
oversampling_factor = 1
dc_suppression = True
modulation_order = 16

resource_block_0 = [FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 5),
                    FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 5)]
ofdm_symbol_0 = FrameResource(prefix_ratio=160/2048, repetitions=100, elements=resource_block_0, prefix_type=PrefixType.CYCLIC)

ofdm_symbol_1 = FrameResource(prefix_ratio=144/2048, repetitions=1, elements=[FrameElement(ElementType.DATA, 1200)],
                              prefix_type=PrefixType.CYCLIC)

resource_block_2 = [FrameElement(ElementType.DATA, 3),
                    FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 5),
                    FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 2)]
ofdm_symbol_2 = FrameResource(prefix_ratio=144/2048, repetitions=100, elements=resource_block_2, prefix_type=PrefixType.CYCLIC)

resources = [ofdm_symbol_0, ofdm_symbol_1, ofdm_symbol_2]

time_slot = FrameSymbolSection(pattern=[0, 1, 1, 1, 2, 1, 1])
structure = [time_slot, time_slot, time_slot, FrameGuardSection(duration=0.5e-3)]

# Create two simulated devices acting as source and sink
tx_device = SimulatedDevice()
rx_device = SimulatedDevice()

# Set up a unidirectional link between both simulated devices
link = SimplexLink(tx_device, rx_device)
link.waveform_generator = OFDMWaveform(subcarrier_spacing=subcarrier_spacing, modulation_order=modulation_order,
                                       num_subcarriers=num_subcarriers, oversampling_factor=oversampling_factor,
                                       dc_suppression=dc_suppression, resources=resources, structure=structure)


# Simulate a channel between the two devices
channel = IdealChannel(tx_device, rx_device)

# Simulate the signal transmission over the channel
transmission = link.transmit()
rx_signal, _, channel_state = channel.propagate(tx_device.transmit())
rx_device.receive(rx_signal)
reception = link.receive()

# Evaluate bit errors during transmission and visualize the received symbol constellation
evaluator = BitErrorEvaluator(link, link)
evaluator.evaluate().plot()
reception.symbols.plot_constellation()
reception.signal.plot()
plt.show()
