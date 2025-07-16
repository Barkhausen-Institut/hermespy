===========
Throughput
===========

.. inheritance-diagram:: hermespy.modem.evaluators.ThroughputEvaluator hermespy.modem.evaluators.ThroughputArtifact hermespy.modem.evaluators.ThroughputEvaluation
   :parts: 1
   :top-classes: hermespy.core.pymonte.evaluation.Evaluator, hermespy.core.monte_carlo.Evaluation, hermespy.core.monte_carlo.Artifact
    
Considering two linked modems denoted by :math:`(\alpha)` and :math:`(\beta)`,
with modem :math:`(\alpha)` transmitting a bit stream

.. math::

   \mathbf{b}_{\mathrm{Tx}}^{(\alpha)} = \left[ b_{\mathrm{Tx}}^{(\alpha,1)}, b_{\mathrm{Tx}}^{(\alpha,2)}, \ldots, b_{\mathrm{Tx}}^{(\alpha,B)} \right]^{\mathsf{T}} \in \lbrace 0, 1 \rbrace^{B}
   
and modem :math:`(\beta)` receiving a bit stream

.. math::

   \mathbf{b}_{\mathrm{Rx}}^{(\beta)} = \left[ b_{\mathrm{Rx}}^{(\beta,1)}, b_{\mathrm{Rx}}^{(\beta,2)}, \ldots, b_{\mathrm{Rx}}^{(\beta,B)} \right]^{\mathsf{T}} \in \lbrace 0, 1 \rbrace^{B}

which can be partitioned into :math:`L` bit block segments of equal length

.. math::

   \mathbf{b}_{\mathrm{Tx}}^{(\alpha)} &= \left[ b_{\mathrm{B,Tx}}^{(\alpha,1)\mathsf{T}}, b_{\mathrm{B,Tx}}^{(\alpha,2)\mathsf{T}}, \ldots, b_{\mathrm{B,Tx}}^{(\alpha,L)\mathsf{T}} \right] \in \lbrace 0, 1 \rbrace^{B} \\
   \mathbf{b}_{\mathrm{Rx}}^{(\beta)} &= \left[ b_{\mathrm{B,Rx}}^{(\beta,1)\mathsf{T}}, b_{\mathrm{B,Rx}}^{(\beta,2)\mathsf{T}}, \ldots, b_{\mathrm{B,Rx}}^{(\beta,L)\mathsf{T}} \right] \in \lbrace 0, 1 \rbrace^{B}

Hermes defines the data througput (DRX) as the exepcted number of block errors between the streams, i.e.,

.. math::

   \mathrm{DRX}^{(\alpha,\beta)} = \mathbb{E} \lbrace \| b_{\mathrm{B,Tx}}^{(\alpha,l)} - b_{\mathrm{B,Rx}}^{(\alpha,l)} \|_2^2 > 0 \rbrace \frac{\mathrm{bit}}{T_\mathrm{F}} \ \text{.}

Note that Hermes currently does not support the concept of Protocal Data Units (DPU).
Hence, the data throughput is estimated based on frame errors.
In practice, the number of bits :math:`B` may differ between transmitter and receiver.
In this case, the shorter bit stream is padded with zeros.

The following minimal examples outlines how to configure this evaluator within the context of a simulation campaign:

.. literalinclude:: ../../scripts/examples/modem_evaluators_throughput.py
   :language: python
   :linenos:
   :lines: 10-33

.. autoclass:: hermespy.modem.evaluators.ThroughputEvaluator

.. autoclass:: hermespy.modem.evaluators.ThroughputArtifact

.. autoclass:: hermespy.modem.evaluators.ThroughputEvaluation

.. footbibliography::
