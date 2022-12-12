import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import Channel
from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, BitErrorEvaluator, OFDMWaveform
from hermespy.modem import FrameResource, FrameElement, ElementType, FrameSymbolSection, FrameGuardSection
from hermespy.modem import GuardType

# model an LTE OFDM link with 20 MHz bandwidth
# 3 data time slots + 1 empty slot are transmitted
subcarrier_spacing = 15e3
num_subcarriers = 1200
oversampling_factor = 2048 / 1200
dc_suppression = True
modulation_order = 16
guard_type = GuardType.CYCLIC_PREFIX
resource_block_0 = [FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 5),
                    FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 5)]
ofdm_symbol_0 = FrameResource(cp_ratio=160/2048, repetitions=100, elements=resource_block_0, guard_type=guard_type)

ofdm_symbol_1 = FrameResource(cp_ratio=144/2048, repetitions=1, elements=[FrameElement(ElementType.DATA, 1200)],
                              guard_type=guard_type)

resource_block_2 = [FrameElement(ElementType.DATA, 3),
                    FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 5),
                    FrameElement(ElementType.REFERENCE), FrameElement(ElementType.DATA, 2)]
ofdm_symbol_2 = FrameResource(cp_ratio=144/2048, repetitions=100, elements=resource_block_2, guard_type=guard_type)

resources = [ofdm_symbol_0, ofdm_symbol_1, ofdm_symbol_2]

time_slot = FrameSymbolSection(pattern=[0, 1, 1, 1, 2, 1, 1])
structure = [time_slot, time_slot, time_slot, FrameGuardSection(duration=0.5e-3)]

# Create two simulated devices acting as source and sink
tx_device = SimulatedDevice()
rx_device = SimulatedDevice()

# Define a transmit operation on the first device
tx_operator = Modem()
tx_operator.waveform_generator = OFDMWaveform(subcarrier_spacing=subcarrier_spacing, modulation_order=modulation_order,
                                              num_subcarriers=num_subcarriers, oversampling_factor=oversampling_factor,
                                              dc_suppression=dc_suppression, resources=resources, structure=structure)
tx_operator.device = tx_device

# Define a receive operation on the second device
rx_operator = Modem()
rx_operator.waveform_generator = OFDMWaveform(subcarrier_spacing=subcarrier_spacing, modulation_order=modulation_order,
                                              num_subcarriers=num_subcarriers, oversampling_factor=oversampling_factor,
                                              dc_suppression=dc_suppression, resources=resources, structure=structure)
rx_operator.device = rx_device

# Simulate a channel between the two devices
channel = Channel(tx_operator.device, rx_operator.device)

# Simulate the signal transmission over the channel
transmission = tx_operator.transmit()
rx_signal, _, channel_state = channel.propagate(tx_device.transmit())
rx_device.receive(rx_signal)
reception = rx_operator.receive()

# Evaluate bit errors during transmission and visualize the received symbol constellation
evaluator = BitErrorEvaluator(tx_operator, rx_operator)
evaluator.evaluate().plot()
reception.symbols.plot_constellation()
plt.show()
