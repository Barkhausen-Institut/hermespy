from hermespy.simulation.simulation import Simulation
from hermespy.simulation.simulated_device import SimulatedDevice
from hermespy.modem import Modem, WaveformGeneratorChirpFsk
from hermespy.modem.evaluators import BitErrorEvaluator


simulation = Simulation()

device = SimulatedDevice()
simulation.add_device(device)

modem = Modem()
modem.waveform_generator = WaveformGeneratorChirpFsk()
modem.device = device

simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.add_dimension('snr', [0.5, 1, 2, 4, 8])
simulation.num_samples = 100
simulation.run()
