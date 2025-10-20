from os.path import join

import matplotlib.pyplot as plt

from hermespy.channel import TDL
from hermespy.hardware_loop import HardwareLoop, PhysicalDeviceDummy, PhysicalScenarioDummy, ReceivedConstellationPlot, DeviceTransmissionPlot, DeviceReceptionPlot
from hermespy.modem import BitErrorEvaluator, SimplexLink, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization

# Create a new hardware loop
hardware_scenario = PhysicalScenarioDummy(seed=42)
hardware_loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](hardware_scenario)

# Add two dedicated devices to the hardware loop, this could be, for example, two USRPs
tx_device = hardware_loop.new_device(carrier_frequency=1e9, oversampling_factor=8)
rx_device = hardware_loop.new_device(carrier_frequency=1e9, oversampling_factor=8)

# Specifiy the channel instance linking the two devices
# Only available for PhysicalScenarioDummy, which is a simulation of hardware behaviour
hardware_scenario.set_channel(tx_device, rx_device, TDL())

# Define a simplex communication link between the two devices
link = SimplexLink()
tx_device.transmitters.add(link)
rx_device.receivers.add(link)

# Configure the waveform to be transmitted over the link
link.waveform = RootRaisedCosineWaveform(
    num_preamble_symbols=10, num_data_symbols=100, roll_off=.9,
)
link.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Add a bit error rate evaluation to the hardware loop
ber = BitErrorEvaluator(link, link)
hardware_loop.add_evaluator(ber)

# Add some runtime visualizations to the hardware loop
hardware_loop.add_plot(DeviceTransmissionPlot(tx_device, 'Tx Signal'))
hardware_loop.add_plot(DeviceReceptionPlot(rx_device, 'Rx Signal'))
hardware_loop.add_plot(ReceivedConstellationPlot(link, 'Rx Constellation'))

# Iterate over the receiving device's SNR and estimate the respective bit error rates
hardware_loop.new_dimension('carrier_frequency', [1e9, 10e9, 100e9], tx_device, rx_device)
hardware_loop.num_drops = 10

hardware_loop.results_dir = hardware_loop.default_results_dir()
hardware_loop.run()

# Replay the recorded dataset
hardware_loop.replay(join(hardware_loop.results_dir, 'drops.h5'))
