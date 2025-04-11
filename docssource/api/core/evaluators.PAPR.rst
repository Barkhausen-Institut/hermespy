======
PAPR
======

.. inheritance-diagram:: hermespy.core.evaluators.PAPREvaluation hermespy.core.evaluators.PAPR hermespy.core.evaluators.PAPRArtifact
   :parts: 1

The *PAPR* evaluator collects information about the Peak-to-Average Power Ratio (*PAPR*) of a signal.
The *PAPR* is defined as the ratio of the peak power to the average power

.. math:: 
   \text{PAPR} = \frac{P_{\mathrm{Peak}}}{P_{\mathrm{Avg}}}

where :math:`P_{peak}` is the peak power and :math:`P_{avg}` is the average power as indicated by the sum of squared complex voltages divided by the number of samples.
Within the context of a simulation, the evaluator can be configured like this:

.. literalinclude:: ../../scripts/examples/core_evaluators_papr.py
   :language: python
   :linenos:
   :lines: 10-26

.. autoclass:: hermespy.core.evaluators.PAPR

.. autoclass:: hermespy.core.evaluators.PAPREvaluation

.. autoclass:: hermespy.core.evaluators.PAPRArtifact

.. footbibliography::
