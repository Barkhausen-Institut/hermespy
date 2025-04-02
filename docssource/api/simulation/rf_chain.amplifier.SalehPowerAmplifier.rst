=====================
Saleh Power Amplifier
=====================

.. inheritance-diagram:: hermespy.simulation.rf_chain.power_amplifier.SalehPowerAmplifier
   :parts: 1

Model of a power amplifier according to :footcite:t:`1981:saleh`.
Implements a saturation characteristic according to

.. math::

   s'(t) = s(t) \cdot A\lbrace s(t) \rbrace e^{\mathrm{j} \Phi\lbrace s(t) \rbrace}

where

.. math::

   A\lbrace s \rbrace =
   \frac{ \alpha_\mathrm{a} \frac{|s|}{s_\mathrm{sat}} }
         { 1 + \beta_\mathrm{a} \frac{|s|^2}{s_\mathrm{sat}^2} }

describes the amplitude model depending on two parameters
:math:`\alpha_\mathrm{a}, \beta_\mathrm{a} \in \mathbb{R}_{+}`
and

.. math::

   \Phi\lbrace s \rbrace =
       \frac{ \alpha_\Phi \frac{|s|}{s_\mathrm{sat}} }
         { 1 + \beta_\Phi \frac{|s|^2}{s_\mathrm{sat}^2} }

describes the phase model depending on
:math:`\alpha_\Phi, \beta_\Phi \in \mathbb{R}`, respectively.

.. autoclass:: hermespy.simulation.rf_chain.power_amplifier.SalehPowerAmplifier

.. footbibliography::
