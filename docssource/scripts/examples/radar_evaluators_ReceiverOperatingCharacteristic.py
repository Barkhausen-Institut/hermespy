import matplotlib.pyplot as plt

from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import Simulation
from hermespy.channel import SingleTargetRadarChannel

# Create a new simulated scenario featuring a single device
simulation = Simulation()
device = simulation.new_device(carrier_frequency=60e9)

# Configure the device to transmit and reveive radar waveforms
radar = Radar(waveform=FMCW())
device.add_dsp(radar)

# Create a new radar channel with a single illuminated target
target = SingleTargetRadarChannel(1, 1., attenuate=True)
simulation.scenario.set_channel(device, device, target)

# Create a new detection probability evaluator
simulation.add_evaluator(ReceiverOperatingCharacteristic(radar, device, device, target))

# Run the simulation
result = simulation.run()
result.plot()
plt.show()
