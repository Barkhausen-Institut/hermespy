# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from hermespy.core import dB, Signal, SignalTransmitter, SignalReceiver
from hermespy.simulation import (
    PerfectCoupling,
    RandomTrigger,
    RfChain,
    Simulation,
    SimulatedDevice,
    SimulatedIdealAntenna,
    SimulatedUniformArray,
    SpecificIsolation, AWGN, N0
)

# Create a new stand-alone simulated device
device = SimulatedDevice()

# Create a new device within a simulation context
simulation = Simulation()
device = simulation.new_device()

# Configure the default carrier frequency at wich waveforms are emitted
device.carrier_frequency = 3.7e9

# Configure the antenna frontend
device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * device.wavelength, [4, 4, 1])

# Configure the default rf-chain model
device.rf_chain = RfChain()

# Configure a transmit-receive isolation model
device.isolation = SpecificIsolation(
    1e-10 * np.ones((device.num_receive_antennas, device.num_transmit_antennas))
)

# Configure a mutual coupling model
device.coupling = PerfectCoupling()

# Configure a trigger model for synchronization
device.trigger_model = RandomTrigger()

# Specify a default sampling rate / bandwidth
device.sampling_rate = 100e6

# Configure a hardware noise model
device.noise_model = AWGN()
device.noise_level = N0(dB(-20))

# Specify the device's postion and orientation in space
device.position = np.array([10., 10., 0.], dtype=np.float_)
device.orientation = np.array([0, .125 * np.pi, 0], dtype=np.float_)

# Specify the device's velocity
device.velocity = np.array([1., 1., 0.], dtype=np.float_)

# Transmit random white noise from the device
transmitter = SignalTransmitter(Signal(
    np.random.normal(size=(device.num_transmit_ports, 100)) +
    1j * np.random.normal(size=(device.num_transmit_ports, 100)),
    device.sampling_rate,
    device.carrier_frequency
))
device.transmitters.add(transmitter)
transmitter.device = device  # Equivalent to the previous line

# Receive a signal without additional processing at the device
receiver = SignalReceiver(100, device.sampling_rate, expected_power=1.)
device.receivers.add(receiver)
receiver.device = device  # Equivalent to the previous line

# Generate a transmission from the device
transmission = device.transmit()

# Receive a signal at the device
impinging_signal = Signal(
    np.random.normal(size=(device.num_transmit_ports, 100)) +
    1j * np.random.normal(size=(device.num_transmit_ports, 100)),
    device.sampling_rate,
    device.carrier_frequency
)
reception = device.receive(impinging_signal)

# Visualize results
transmission.mixed_signal.plot(title='Transmitted Waveform')
reception.operator_receptions[0].signal.plot(title='Received Waveform')
plt.show()
