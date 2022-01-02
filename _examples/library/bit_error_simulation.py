from hermespy.simulation.simulation import Simulation
from hermespy.simulation.simulated_device import SimulatedDevice
from hermespy.modem import Modem, WaveformGeneratorChirpFsk, WaveformGeneratorPskQam
from hermespy.modem.evaluators import BitErrorEvaluator, BlockErrorEvaluator, FrameErrorEvaluator


simulation = Simulation()

device = SimulatedDevice()
simulation.add_device(device)

modem = Modem()
modem.waveform_generator = WaveformGeneratorPskQam()
modem.device = device

simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.add_evaluator(BlockErrorEvaluator(modem, modem))
simulation.add_evaluator(FrameErrorEvaluator(modem, modem))

simulation.add_dimension('snr', [0.5, 1, 2, 4, 8, 16])
simulation.num_samples = 1000
simulation.run()
