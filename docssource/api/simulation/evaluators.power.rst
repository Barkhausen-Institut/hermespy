===========
Block Power
===========

.. inheritance-diagram:: hermespy.simulation.evaluators.power.PowerConsumptionEvaluator hermespy.simulation.evaluators.power.PowerConsumptionEvaluation hermespy.simulation.evaluators.power.DCPowerConsumptionEvaluator hermespy.simulation.evaluators.power.DCPowerConsumptionEvaluation
   :parts: 1

The power consumption evaluator relates the signal power a radio-frequency block
emits to the power supplied to it, both as signal power at its input and as direct
current drawn from its supply.
The resulting total efficiency

.. math::

   \eta = \frac{P_{\mathrm{out}}}{P_{\mathrm{in}} + P_{\mathrm{DC}}}

approaches the conversion efficiency of the modeled component when the block is
driven hard, and tends towards zero when it is driven softly, since an
:class:`ActiveRFBlock<hermespy.simulation.rf.block.ActiveRFBlock>` continues to draw
its quiescent power while emitting almost nothing.

The direct current contribution :math:`P_{\mathrm{DC}}` is supplied by the block's
:class:`DCPowerModel<hermespy.simulation.rf.power.DCPowerModel>`.
It is zero for a
:class:`PassiveRFBlock<hermespy.simulation.rf.block.PassiveRFBlock>`, whose efficiency
therefore describes its insertion loss alone.

Samples during which neither the input nor the output of the block carries any power
are excluded from the average, so that the metric describes the block itself rather
than the duty cycle of the processed signal.
As a consequence, direct current power drawn while the block is idle does not enter
the efficiency.

Efficiency is not meaningful for every block. A
:class:`DAC<hermespy.simulation.rf.blocks.ad.DAC>` emits no radio-frequency signal
whose power could be related to its supply, and quantization leaves the signal power
of an :class:`ADC<hermespy.simulation.rf.blocks.ad.ADC>` almost unchanged, so that its
efficiency approaches unity regardless of the configured resolution. For those blocks
:class:`DCPowerConsumptionEvaluator<hermespy.simulation.evaluators.power.DCPowerConsumptionEvaluator>`
reports the drawn supply power itself, averaged over every sample rather than only
over the samples during which the block is driven.

Each active block declares what it draws from the supply, and the evaluator
subscribes to a single block of the chain.

.. literalinclude:: ../../scripts/examples/simulation_evaluation_power.py
   :language: python
   :linenos:
   :lines: 29-46,75-81

.. autoclass:: hermespy.simulation.evaluators.power.PowerConsumptionEvaluator

.. autoclass:: hermespy.simulation.evaluators.power.PowerConsumptionEvaluation

.. autoclass:: hermespy.simulation.evaluators.power.DCPowerConsumptionEvaluator

.. autoclass:: hermespy.simulation.evaluators.power.DCPowerConsumptionEvaluation

.. footbibliography::
