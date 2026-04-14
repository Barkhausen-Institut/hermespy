==========
Amplifiers
==========

.. inheritance-diagram:: hermespy.simulation.rf.blocks.amps.PowerAmplifier hermespy.simulation.rf.blocks.amps.ClippingPowerAmplifier hermespy.simulation.rf.blocks.amps.RappPowerAmplifier hermespy.simulation.rf.blocks.amps.SalehPowerAmplifier hermespy.simulation.rf.blocks.amps.CustomPowerAmplifier
   :parts: 1

Radio-frequency (RF) amplifiers are blocks increasing the power (or equivalently the amplitude) of an input signals passing
through the block.
In the process, they introduce hardmonic frequency components due to their non-linear amplification characteristics,
phase shifts and thermal noise.
Deployed in high frequency applications they may additionally introduce memory effects due to their internal circuitry.

This module provides various numerical models of RF amplifiers implementing the basic relationship

.. math::

   s'(t) = f\lbrace s(t), t, \tau \rbrace \in \mathbb{C}

between an electromagnetic complex-valued input signal :math:`s(t) \in \mathbb{C}` and the respective output signal :math:`s'(t) \in \mathbb{C}`
with respective to a past delay component :math:`\tau` and time variable :math:`t`.
Note that, for memoryless models, this function reduces to

.. math::

   s'(t) = f\lbrace s(t), t \rbrace \in \mathbb{C} \quad \text{.}


---------------
Ideal Amplifier
---------------

The power amplifier is usually the last stage in a transmitting device's radio frequency chain before the antenna.
This base model is HermesPy's default power amplifier model, which is a linear amplifier with a constant gain and
therfore no distortion

.. math::

   s'(t) = s(t) \text{,}

which may be overwritten by classes inheriting from this base.

.. autoclass:: hermespy.simulation.rf.blocks.amps.PowerAmplifier


------------------
Clipping Amplifier
------------------

Model of a power amplifier driven into saturation.
Complex signal samples with amplitudes above :math:`s_\mathrm{sat} \in \mathbb{R}` will be clipped
to the maximum amplitude value.
In case of clipping, the respective sample angles will be preserved, so that

.. math::

   s'(t) = \frac{s(t)}{|s(t)|} \cdot \min{(|s(t)|, s_\mathrm{sat})} \text{.}

.. autoclass:: hermespy.simulation.rf.blocks.amps.ClippingPowerAmplifier


---------------
Saleh Amplifier
---------------

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

.. autoclass:: hermespy.simulation.rf.blocks.amps.SalehPowerAmplifier


--------------
Rapp Amplifier
--------------

Model of a power amplifier according to :footcite:t:`1991:rapp`.
Implements a saturation characteristic according to

.. math::

   s'(t) = s(t) \cdot \left( 1 + \left( \frac{|s(t)|}{s_\mathrm{sat}} \right)^{2p_\mathrm{Rapp}} \right)^{-\frac{1}{2p_\mathrm{Rapp}}} \text{,}

where :math:`p_\mathrm{Rapp} \in \lbrace x \in \mathbb{R} | x \geq 1 \rbrace` denotes the smoothness factor of the saturation curve.

.. autoclass:: hermespy.simulation.rf.blocks.amps.RappPowerAmplifier


----------------
Custom Amplifier
----------------

Fully customizable pwoer amplification model.
The users may define their own gain and phase characteristics.

.. autoclass:: hermespy.simulation.rf.blocks.amps.CustomPowerAmplifier


.. footbibliography::