=========================
Clipping Power Amplifier
=========================

.. inheritance-diagram:: hermespy.simulation.rf_chain.power_amplifier.ClippingPowerAmplifier
   :parts: 1

Model of a power amplifier driven into saturation.
Complex signal samples with amplitudes above :math:`s_\mathrm{sat} \in \mathbb{R}` will be clipped
to the maximum amplitude value.
In case of clipping, the respective sample angles will be preserved, so that

.. math::

   s'(t) = \frac{s(t)}{|s(t)|} \cdot \min{(|s(t)|, s_\mathrm{sat})} \text{.}

.. autoclass:: hermespy.simulation.rf_chain.power_amplifier.ClippingPowerAmplifier

.. footbibliography::
