
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi, speed_of_light
from ray import init as ray_init

from hermespy.beamforming import BeamformingTransmitter, ConventionalBeamformer, DeviceFocus, SphericalFocus
from hermespy.core import AntennaMode, ReceivedPowerEvaluator, SamplePoint, Signal, SignalReceiver, Transformation, ValueType
from hermespy.channel import StreetCanyonLineOfSight
from hermespy.simulation import SimulatedCustomArray, SimulatedIdealAntenna, SimulatedAntennaPort, Simulation


sampling_rate = 1e9
carrier_frequency = 70e9
wavelength = speed_of_light / carrier_frequency

uniform_ports = [
    SimulatedAntennaPort(
        [SimulatedIdealAntenna(AntennaMode.TX, Transformation.From_Translation(np.array([0, 0, 0])))],
        Transformation.From_Translation(np.array([.5 * wavelength * x, .5 * wavelength * y, 0]))
    ) for x, y in product(range(4), range(4))
]

hybrid_ports = [
    SimulatedAntennaPort(
        [SimulatedIdealAntenna(AntennaMode.TX, Transformation.From_Translation(np.array([0, .5 * wavelength * y, 0]))) for y in range(4)],
        Transformation.From_Translation(np.array([.5 * wavelength * x, 0, 0]))
    ) for x in range(4)
]

hybrid_array = SimulatedCustomArray(hybrid_ports)
uniform_array = SimulatedCustomArray(uniform_ports)
#array.plot_topology()

print(f"Tx Ports: {uniform_array.num_transmit_ports}")
print(f"Rx Ports: {uniform_array.num_receive_ports}")
print(f"Tx Antennas: {uniform_array.num_transmit_antennas}")
print(f"Rx Antennas: {uniform_array.num_receive_antennas}")

#ray_init(local_mode=True)
simulation = Simulation(seed=42)

base_station_device = simulation.new_device(
    antennas=uniform_array,
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([0, 0, 0])),
)
user_equipment_device = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([20., 20., 100.])),
)
simulation.set_channel(
    base_station_device,
    user_equipment_device,
    StreetCanyonLineOfSight(),
)

test_signal = Signal.Create(np.outer(np.ones(1), np.exp(2j * pi * .25 * np.arange(100))), sampling_rate, carrier_frequency)
beamformer = ConventionalBeamformer()
base_station_transmitter = BeamformingTransmitter(test_signal, beamformer)
base_station_transmitter.device = base_station_device

user_equipment_receiver = SignalReceiver(120, sampling_rate, test_signal.power[0])
user_equipment_receiver.device = user_equipment_device
simulation.add_evaluator(ReceivedPowerEvaluator(user_equipment_receiver))

off_target_focus = SphericalFocus(-.75 * pi, .4 * pi)
beamformer.transmit_focus = off_target_focus
uniform_array.plot_pattern(carrier_frequency, AntennaMode.TX, beamformer)
simulation.scenario.seed = 42
spherical_drop = simulation.scenario.drop()
user_equipment_receiver.reception.signal.plot(title='Off-Target UE Reception')

on_target_focus = DeviceFocus(user_equipment_device)
beamformer.transmit_focus = on_target_focus
uniform_array.plot_pattern(carrier_frequency, AntennaMode.TX, beamformer)
simulation.scenario.seed = 42
device_drop = simulation.scenario.drop()
user_equipment_receiver.reception.signal.plot(title='On-Target UE Reception')

focus_dimension = simulation.new_dimension(
    'transmit_focus',
    [
        SamplePoint(off_target_focus, 'Off-Target'),
        SamplePoint(on_target_focus, 'On-Target'),
    ],
    beamformer,
    title='Beam Focus',
)

simulation.new_dimension(
    'antennas',
    [
        SamplePoint(uniform_array, 'Uniform'),
        SamplePoint(hybrid_array, 'Hybrid'),
    ],
    base_station_device,
    title='Array',
)

simulation.num_drops = 200
array_comparison = simulation.run()

plt.show()
