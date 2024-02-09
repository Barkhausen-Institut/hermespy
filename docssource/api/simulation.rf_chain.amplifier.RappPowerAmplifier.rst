====================
Rapp Power Amplifier
====================

.. inheritance-diagram:: hermespy.simulation.rf_chain.power_amplifier.RappPowerAmplifier
   :parts: 1

Model of a power amplifier according to :footcite:t:`1991:rapp`.
Implements a saturation characteristic according to

.. math::

   s'(t) = s(t) \cdot \left( 1 + \left( \frac{|s(t)|}{s_\mathrm{sat}} \right)^{2p_\mathrm{Rapp}} \right)^{-\frac{1}{2p_\mathrm{Rapp}}} \text{,}

where :math:`p_\mathrm{Rapp} \in \lbrace x \in \mathbb{R} | x \geq 1 \rbrace` denotes the smoothness factor of the saturation curve.

.. autoclass:: hermespy.simulation.rf_chain.power_amplifier.RappPowerAmplifier

.. footbibliography::
