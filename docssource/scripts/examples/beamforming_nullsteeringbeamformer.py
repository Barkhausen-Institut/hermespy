from itertools import product
from copy import deepcopy
import numpy as np
from scipy.constants import speed_of_light

from hermespy.core import Transformation, ReceivePowerEvaluator
from hermespy.simulation import Simulation, SimulatedIdealAntenna, DeviceFocus, SimulatedUniformArray
from hermespy.beamforming import NullSteeringBeamformer
from hermespy.channel import SpatialDelayChannel
from hermespy.modem import TransmittingModem, ReceivingModem, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierCorrelationSynchronization


sampling_rate = 1e6
carrier_frequency = 70e9
wavelength = speed_of_light / carrier_frequency

# Initialize a new simulation
simulation = Simulation(seed=42)

# Create a new device and assign it the antenna array
base_station_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna, .5 * wavelength, (5, 5, 1)),
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
base_station_device.transmit_coding[0] = beamformer

base_station_transmitter = TransmittingModem(waveform=deepcopy(waveform))
base_station_device.add_dsp(base_station_transmitter)

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
ue_receiver_1 = ReceivingModem(waveform=deepcopy(waveform))
ue_receiver_2 = ReceivingModem(waveform=deepcopy(waveform))
ue_receiver_3 = ReceivingModem(waveform=deepcopy(waveform))
user_equipment_device_1.add_dsp(ue_receiver_1)
user_equipment_device_2.add_dsp(ue_receiver_2)
user_equipment_device_3.add_dsp(ue_receiver_3)

# Focus the base station's main lobe on the desired user equipment and nulls on the others
beamformer.transmit_focus = [
    DeviceFocus(user_equipment_device_1),  # Focus on UE1
    DeviceFocus(user_equipment_device_2),  # Null on UE2
    DeviceFocus(user_equipment_device_3),  # Null on UE3
]

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
simulation.add_evaluator(ReceivePowerEvaluator(ue_receiver_1))
simulation.add_evaluator(ReceivePowerEvaluator(ue_receiver_2))
simulation.add_evaluator(ReceivePowerEvaluator(ue_receiver_3))
result = simulation.run()
