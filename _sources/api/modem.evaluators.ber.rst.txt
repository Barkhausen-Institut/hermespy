===============
Bit Error Rate
===============

.. inheritance-diagram:: hermespy.modem.evaluators.BitErrorEvaluator hermespy.modem.evaluators.BitErrorArtifact hermespy.modem.evaluators.BitErrorEvaluation
   :parts: 1
   :top-classes: hermespy.core.monte_carlo.Evaluator, hermespy.core.monte_carlo.Evaluation, hermespy.core.monte_carlo.Artifact

Considering two linked modems denoted by :math:`(\alpha)` and :math:`(\beta)`,
with modem :math:`(\alpha)` transmitting a bit stream

.. math::

   \mathbf{b}_{\mathrm{Tx}}^{(\alpha)} = \left[ b_{\mathrm{Tx}}^{(\alpha,1)}, b_{\mathrm{Tx}}^{(\alpha,2)}, \ldots, b_{\mathrm{Tx}}^{(\alpha,B)} \right]^{\mathsf{T}} \in \lbrace 0, 1 \rbrace^{B}
   
and modem :math:`(\beta)` receiving a bit stream

.. math::

   \mathbf{b}_{\mathrm{Rx}}^{(\beta)} = \left[ b_{\mathrm{Rx}}^{(\beta,1)}, b_{\mathrm{rx}}^{(\beta,2)}, \ldots, b_{\mathrm{Rx}}^{(\beta,B)} \right]^{\mathsf{T}} \in \lbrace 0, 1 \rbrace^{B}

Hermes defines the bit error rate (BER) as the average number of bit errors between the streams

.. math::

   \mathrm{BER}^{(\alpha,\beta)} = \frac{ \lVert \mathbf{b}_{\mathrm{Tx}}^{(\alpha)} - \mathbf{b}_{\mathrm{Rx}}^{(\beta)} \rVert_2^2 }{ B} \ \text{.}

In practice, the number of bits :math:`B` may differ between transmitter and receiver.
In this case, the shorter bit stream is padded with zeros.

The following minimal examples outlines how to configure this evaluator
within the context of a simulation campaign:

.. literalinclude:: ../scripts/examples/modem_evaluators_ber.py
   :language: python
   :linenos:
   :lines: 10-33

.. autoclass:: hermespy.modem.evaluators.BitErrorEvaluator

.. autoclass:: hermespy.modem.evaluators.BitErrorArtifact

.. autoclass:: hermespy.modem.evaluators.BitErrorEvaluation

.. footbibliography::
