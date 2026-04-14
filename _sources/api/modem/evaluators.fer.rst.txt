=================
Frame Error Rate
=================

.. inheritance-diagram:: hermespy.modem.evaluators.FrameErrorEvaluator hermespy.modem.evaluators.FrameErrorArtifact hermespy.modem.evaluators.FrameErrorEvaluation
   :parts: 1
   :top-classes: hermespy.core.pymonte.evaluation.Evaluator, hermespy.core.monte_carlo.Evaluation, hermespy.core.monte_carlo.Artifact
    
Considering two linked modems denoted by :math:`(\alpha)` and :math:`(\beta)`,
with modem :math:`(\alpha)` transmitting a bit stream

.. math::

   \mathbf{b}_{\mathrm{Tx}}^{(\alpha)} = \left[ b_{\mathrm{Tx}}^{(\alpha,1)}, b_{\mathrm{Tx}}^{(\alpha,2)}, \ldots, b_{\mathrm{Tx}}^{(\alpha,B)} \right]^{\mathsf{T}} \in \lbrace 0, 1 \rbrace^{B}
   
and modem :math:`(\beta)` receiving a bit stream

.. math::

   \mathbf{b}_{\mathrm{Rx}}^{(\beta)} = \left[ b_{\mathrm{Rx}}^{(\beta,1)}, b_{\mathrm{Rx}}^{(\beta,2)}, \ldots, b_{\mathrm{Rx}}^{(\beta,B)} \right]^{\mathsf{T}} \in \lbrace 0, 1 \rbrace^{B}

which can be partitioned into :math:`L` frame segments of equal length

.. math::

   \mathbf{b}_{\mathrm{Tx}}^{(\alpha)} &= \left[ b_{\mathrm{F,Tx}}^{(\alpha,1)\mathsf{T}}, b_{\mathrm{F,Tx}}^{(\alpha,2)\mathsf{T}}, \ldots, b_{\mathrm{F,Tx}}^{(\alpha,L)\mathsf{T}} \right] \in \lbrace 0, 1 \rbrace^{B} \\
   \mathbf{b}_{\mathrm{Rx}}^{(\beta)} &= \left[ b_{\mathrm{F,Rx}}^{(\beta,1)\mathsf{T}}, b_{\mathrm{F,Rx}}^{(\beta,2)\mathsf{T}}, \ldots, b_{\mathrm{F,Rx}}^{(\beta,L)\mathsf{T}} \right] \in \lbrace 0, 1 \rbrace^{B}

Hermes defines the frame error rate (FER) as the exepcted number of block errors between the streams, i.e.,

.. math::

   \mathrm{FER}^{(\alpha,\beta)} = \mathbb{E} \lbrace \| b_{\mathrm{F,Tx}}^{(\alpha,l)} - b_{\mathrm{F,Rx}}^{(\alpha,l)} \|_2^2 > 0 \rbrace \ \text{.}

In practice, the number of bits :math:`B` may differ between transmitter and receiver.
In this case, the shorter bit stream is padded with zeros.

The following minimal examples outlines how to configure this evaluator within the context of a simulation campaign:

.. literalinclude:: ../../scripts/examples/modem_evaluators_fer.py
   :language: python
   :linenos:
   :lines: 10-33

.. autoclass:: hermespy.modem.evaluators.FrameErrorEvaluator

.. autoclass:: hermespy.modem.evaluators.FrameErrorArtifact

.. autoclass:: hermespy.modem.evaluators.FrameErrorEvaluation

.. footbibliography::
