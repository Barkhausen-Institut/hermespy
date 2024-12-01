from itertools import product
from copy import deepcopy
import numpy as np
from scipy.constants import speed_of_light

from hermespy.core import AntennaMode, Transformation, ReceivePowerEvaluator
from hermespy.simulation import Simulation, SimulatedIdealAntenna, DeviceFocus, SimulatedCustomArray
from hermespy.beamforming import NullSteeringBeamformer
from hermespy.channel import SpatialDelayChannel
from hermespy.modem import TransmittingModem, ReceivingModem, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierCorrelationSynchronization


sampling_rate = 1e6
carrier_frequency = 70e9
wavelength = speed_of_light / carrier_frequency

# Configure an antenna array with custom coordinates
uniform_array = SimulatedCustomArray()
for x, y in product(range(4), range(4)):
    uniform_array.add_antenna(SimulatedIdealAntenna(
        mode=AntennaMode.TX,
        pose=Transformation.From_Translation(np.array([.5 * wavelength * x, .5 * wavelength * y, 0])),
    ))

# Initialize a new simulation
simulation = Simulation(seed=42)

# Create a new device and assign it the antenna array
base_station_device = simulation.new_device(
    antennas=uniform_array,
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([0, 0, 0])),
)

# Configure a probong signal to be transmitted from the base station
waveform = RootRaisedCosineWaveform(
    symbol_rate=sampling_rate//2,
    oversampling_factor=2,
    num_preamble_symbols=32,
    num_data_symbols=128,
    roll_off=.9,
)
waveform.synchronization = SingleCarrierCorrelationSynchronization()
waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Configure the base station device to transmit the beamformed probing signal
beamformer = NullSteeringBeamformer()

base_station_transmitter = TransmittingModem(waveform=deepcopy(waveform),device=base_station_device)
base_station_transmitter.device = base_station_device
base_station_transmitter.transmit_stream_coding[0] = beamformer

# Create three simulated devices representing the user equipments (Since nullsteeringbeamformer has 3 focus points)
user_equipment_device_1 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([100., 100., 100.])),
)

user_equipment_device_2 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([-100., 100., 100.])),
)

user_equipment_device_3 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([100., -100., 100.])),
)

# Configure the user equipments to receive the signal
user_equipment_transmitter_1 = ReceivingModem(waveform=deepcopy(waveform), device=user_equipment_device_1)
user_equipment_transmitter_2 = ReceivingModem(waveform=deepcopy(waveform), device=user_equipment_device_2)
user_equipment_transmitter_3 = ReceivingModem(waveform=deepcopy(waveform), device=user_equipment_device_3)

# Focus the base station's main lobe on the desired user equipment and nulls on the others
beamformer.transmit_focus = [DeviceFocus(base_station_device, user_equipment_device_1),
                            DeviceFocus(base_station_device, user_equipment_device_2),
                            DeviceFocus(base_station_device, user_equipment_device_3)]

# Configure a channel between base station and the UEs
simulation.set_channel(
    base_station_device,
    user_equipment_device_1,
    SpatialDelayChannel(model_propagation_loss=False),
)

simulation.set_channel(
    base_station_device,
    user_equipment_device_2,
    SpatialDelayChannel(model_propagation_loss=False),
)

simulation.set_channel(
    base_station_device,
    user_equipment_device_3,
    SpatialDelayChannel(model_propagation_loss=False),
)

# Configure a simulation scenario and analyze the power received by the 3 UEs by adding an evaluator.
simulation.scenario.channel(base_station_device, base_station_device).gain = 0.0
simulation.scenario.drop()
simulation.add_evaluator(ReceivePowerEvaluator(user_equipment_transmitter_1))
simulation.add_evaluator(ReceivePowerEvaluator(user_equipment_transmitter_2))
simulation.add_evaluator(ReceivePowerEvaluator(user_equipment_transmitter_3))
simulation.run()
