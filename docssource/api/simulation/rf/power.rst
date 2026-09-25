====================
DC Power Consumption
====================

.. inheritance-diagram:: hermespy.simulation.rf.power.DCPowerModel hermespy.simulation.rf.power.NoDCPowerModel hermespy.simulation.rf.power.ConstantDCPowerModel hermespy.simulation.rf.power.SampledDCPowerModel hermespy.simulation.rf.power.WaldenADCPowerModel
   :parts: 1

Direct current power consumption models describe the amount of power an
:class:`ActiveRFBlock<hermespy.simulation.rf.block.ActiveRFBlock>` draws from its supply
while processing signals.
They are the direct-current counterpart of
:class:`NoiseLevel<hermespy.simulation.rf.noise.level.NoiseLevel>`,
which describes the noise a block contributes to the processed signal.

Which model is appropriate depends on how strongly the consumption of the modeled
component varies with the drive level.
Amplifiers biased at a fixed operating point draw a nearly constant current regardless
of the processed signal and are well represented by a
:class:`ConstantDCPowerModel<hermespy.simulation.rf.power.ConstantDCPowerModel>`.
Components whose consumption follows the signal envelope require the tabulated
:class:`SampledDCPowerModel<hermespy.simulation.rf.power.SampledDCPowerModel>` instead,
which interpolates measurements taken across the operating range.
Analog-to-digital converters are the exception to both: their consumption follows
from the configured resolution and bandwidth rather than from a measurement, and is
captured by the
:class:`WaldenADCPowerModel<hermespy.simulation.rf.power.WaldenADCPowerModel>`.

A model is handed to the block it describes, which draws on it whenever a signal
propagates through the chain.

.. literalinclude:: ../../../scripts/examples/simulation_evaluation_power.py
   :language: python
   :linenos:
   :lines: 51-58

.. autoclass:: hermespy.simulation.rf.power.DCPowerModel

.. autoclass:: hermespy.simulation.rf.power.NoDCPowerModel

.. autoclass:: hermespy.simulation.rf.power.ConstantDCPowerModel

.. autoclass:: hermespy.simulation.rf.power.SampledDCPowerModel

.. autoclass:: hermespy.simulation.rf.power.WaldenADCPowerModel

.. footbibliography::