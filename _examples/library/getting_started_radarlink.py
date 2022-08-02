import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import Channel, RadarChannel
from hermespy.simulation import SimulatedDevice
from hermespy.radar import Radar, DetectionProbEvaluator, ReceiverOperatingCharacteristic, FMCW

# Create two simulated devices acting as source and sink
device = SimulatedDevice(carrier_frequency=10e9)
device_h0 = SimulatedDevice(carrier_frequency=10e9)

# Define a transmit operation on the first device
tx_operator = Radar()
tx_operator.waveform = FMCW()
tx_operator.device = device

tx_operator_h0 = Radar()
tx_operator_h0.waveform = FMCW()
tx_operator_h0.device = device_h0

# Define a receive operation on the second device
rx_operator = Radar()
rx_operator.waveform = FMCW()
rx_operator.device = device

rx_operator_h0 = Radar()
rx_operator_h0.waveform = FMCW()
rx_operator_h0.device = device_h0

# Simulate a channel between the two devices
channel = RadarChannel(target_range=50, radar_cross_section=1, target_exists=True)
channel.transmitter = device
channel.receiver = device

channel_h0 = RadarChannel(target_range=50, radar_cross_section=1, target_exists=False)
channel_h0.transmitter = device_h0
channel_h0.receiver = device_h0

# Simulate the signal transmission over the channel
tx_signal, = tx_operator.transmit()
rx_signal, _, channel_state = channel.propagate(tx_signal)
device.receive(rx_signal)
rx_operator.receive()


tx_signal_h0, = tx_operator_h0.transmit()
rx_signal_h0, _, channel_state_h0 = channel_h0.propagate(tx_signal_h0)
device_h0.receive(rx_signal_h0)
rx_operator_h0.receive()

evaluator1 = DetectionProbEvaluator(rx_operator)
detection = evaluator1.evaluate().to_scalar()

print(f"Detection = {detection}")

evaluator2 = ReceiverOperatingCharacteristic(rx_operator, rx_operator_h0)
detection_values = evaluator2.evaluate()

print(f"Values = ({detection_values[0].to_scalar()}, {detection_values[1].to_scalar()})")
