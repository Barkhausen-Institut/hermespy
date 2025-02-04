from scipy.constants import speed_of_light

sampling_rate = 1e6
carrier_frequency = 70e9
wavelength = speed_of_light / carrier_frequency

# Initialize a new simulation
from hermespy.simulation import Simulation
simulation = Simulation(seed=42)

# Create a new device and assign it the antenna array
from hermespy.simulation import SimulatedUniformArray, SimulatedIdealAntenna
from hermespy.core import Transformation
import numpy as np

base_station_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna, .5 * wavelength, (5, 5, 1)),
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([0, 0, 0])),
)

# Configure a probing signal to be transmitted from the base station
from hermespy.modem import RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierCorrelationSynchronization

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
from hermespy.beamforming import NullSteeringBeamformer
from hermespy.modem import TransmittingModem
from copy import deepcopy

beamformer = NullSteeringBeamformer()
base_station_device.transmit_coding[0] = beamformer

base_station_transmitter = TransmittingModem(waveform=deepcopy(waveform))
base_station_device.add_dsp(base_station_transmitter)

# Create three simulated devices representing the user equipments (Since nullsteeringbeamformer has 3 focus points)
user_equipment_device_1 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([200., 200., 100.])),
)
user_equipment_device_2 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([200., -200., 100.])),
)
user_equipment_device_3 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([-400., 200., 100.])),
)

# Configure the user equipments to receive the signal
from hermespy.modem import ReceivingModem

ue_receiver_1 = ReceivingModem(waveform=deepcopy(waveform))
ue_receiver_2 = ReceivingModem(waveform=deepcopy(waveform))
ue_receiver_3 = ReceivingModem(waveform=deepcopy(waveform))
user_equipment_device_1.add_dsp(ue_receiver_1)
user_equipment_device_2.add_dsp(ue_receiver_2)
user_equipment_device_3.add_dsp(ue_receiver_3)

# Focus the base station's main lobe on the desired user equipment and nulls on the others
from hermespy.simulation import DeviceFocus

beamformer.transmit_focus = [
    DeviceFocus(user_equipment_device_1),  # Focus on UE1
    DeviceFocus(user_equipment_device_2),  # Null on UE2
    DeviceFocus(user_equipment_device_3),  # Null on UE3
]

# Configure a channel between base station and the UEs
from hermespy.channel import SpatialDelayChannel

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

# Run the simulation and analyze the power received by the 3 UEs by adding an evaluator.
from hermespy.core import ReceivePowerEvaluator

simulation.add_evaluator(ReceivePowerEvaluator(ue_receiver_1))
simulation.add_evaluator(ReceivePowerEvaluator(ue_receiver_2))
simulation.add_evaluator(ReceivePowerEvaluator(ue_receiver_3))

# Creating a new dimension to dynamically switch the focus of the beamformer during the simulation campaign.
simulation.new_dimension(
    'transmit_focus',
    [
        [DeviceFocus(user_equipment_device_1), DeviceFocus(user_equipment_device_2), DeviceFocus(user_equipment_device_3)],
        [DeviceFocus(user_equipment_device_2), DeviceFocus(user_equipment_device_3), DeviceFocus(user_equipment_device_1)],
        [DeviceFocus(user_equipment_device_3), DeviceFocus(user_equipment_device_1), DeviceFocus(user_equipment_device_2)],
    ],
    beamformer
)
result = simulation.run()
