===============
Power Amplifier
===============

.. inheritance-diagram:: hermespy.simulation.rf_chain.power_amplifier.PowerAmplifier
   :parts: 1

The power amplifier is usually the last stage in a transmitting device's radio frequency chain before the antenna.
This base model is HermesPy's default power amplifier model, which is a linear amplifier with a constant gain and
therfore no distortion

.. math::

   s'(t) = s(t) \text{,}

which may be overwritten by classes inheriting from this base.

.. autoclass:: hermespy.simulation.rf_chain.power_amplifier.PowerAmplifier

.. footbibliography::
