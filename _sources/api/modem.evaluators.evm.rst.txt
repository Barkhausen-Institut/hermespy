======================
Error Vector Magnitude
======================

.. inheritance-diagram:: hermespy.modem.evaluators.ConstellationEVM hermespy.modem.evaluators.EVMArtifact hermespy.modem.evaluators.EVMEvaluation
   :parts: 1
   :top-classes: hermespy.core.monte_carlo.Evaluator, hermespy.core.monte_carlo.Evaluation, hermespy.core.monte_carlo.Artifact

Considering two linked modems denoted by :math:`(\alpha)` and :math:`(\beta)`,
with modem :math:`(\alpha)` transmitting a symbol sequence

.. math::

   \mathbf{s}_{\mathrm{Tx}}^{(\alpha)} = \left[ s_{\mathrm{Tx}}^{(\alpha,1)}, s_{\mathrm{Tx}}^{(\alpha,2)}, \ldots, s_{\mathrm{Tx}}^{(\alpha,B)} \right]^{\mathsf{T}} \in \mathbb{C}^{S}
   
and modem :math:`(\beta)` receiving a decoded symbol sequence

.. math::

   \mathbf{s}_{\mathrm{Rx}}^{(\beta)} = \left[ s_{\mathrm{Rx}}^{(\beta,1)}, s_{\mathrm{rx}}^{(\beta,2)}, \ldots, s_{\mathrm{Rx}}^{(\beta,B)} \right]^{\mathsf{T}} \in \mathbb{C}^{S}

Hermes defines the symbol Error Vector Magnitude (EVM) as the root mean square (RMS) of the difference between the transmitted and received symbols

.. math::

   \mathrm{EVM}^{(\alpha,\beta)} = \sqrt{\frac{ \lVert \mathbf{s}_{\mathrm{Tx}}^{(\alpha)} - \mathbf{s}_{\mathrm{Rx}}^{(\beta)} \rVert_2^2 }{S}} \ \text{.}

In practice, the number of symbols :math:`S` may differ between transmitter and receiver.
In this case, the longer sequence is truncated to the length of the shorter sequence.

The following minimal examples outlines how to configure this evaluator
within the context of a simulation campaign:

.. literalinclude:: ../scripts/examples/modem_evaluators_evm.py
   :language: python
   :linenos:
   :lines: 10-33

.. autoclass:: hermespy.modem.evaluators.ConstellationEVM

.. autoclass:: hermespy.modem.evaluators.EVMArtifact

.. autoclass:: hermespy.modem.evaluators.EVMEvaluation

.. footbibliography::
