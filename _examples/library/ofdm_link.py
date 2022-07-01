import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from hermespy.simulation import SimulatedDevice
from hermespy.channel import IndoorFactoryNoLineOfSight
from hermespy.modem import Modem, WaveformGeneratorOfdm, PilotSection, Symbols

device_alice = SimulatedDevice(position=np.array([0, 0, 0]))
device_bob = SimulatedDevice(position=np.array([10, 10, 0]))

num_subcarriers = 1200
pilot_symbols = Symbols(np.exp(1j * pi * np.random.uniform(0, 2, num_subcarriers)))

modem_alice = Modem()
modem_alice.device = device_alice
modem_alice.waveform_generator = WaveformGeneratorOfdm(oversampling_factor=2, num_subcarriers=num_subcarriers, structure=[PilotSection(pilot_symbols)])

modem_bob = Modem()
modem_bob.device = device_bob
modem_bob.waveform_generator = WaveformGeneratorOfdm(oversampling_factor=2, num_subcarriers=num_subcarriers, structure=[PilotSection(pilot_symbols)])

channel = IndoorFactoryNoLineOfSight(100, 100**2, transmitter=device_alice, receiver=device_bob)

tx_alice, _ , _ = modem_alice.transmit()
tx_bob, _ , _  = modem_bob.transmit()

tx_device_alice = device_alice.transmit()
tx_device_bob = device_bob.transmit()

rx_device_alice, rx_device_bob, _ = channel.propagate(tx_device_alice, tx_device_bob)

device_alice.receive(rx_device_alice)
device_bob.receive(rx_device_bob)

rx_alice, _, _ = modem_alice.receive()
rx_bob, _, _ = modem_bob.receive()

tx_alice.plot('Alice Tx')
tx_bob.plot('Bob Tx')
rx_alice.plot('Alice Rx')
rx_bob.plot('Bob Rx')

plt.show()
