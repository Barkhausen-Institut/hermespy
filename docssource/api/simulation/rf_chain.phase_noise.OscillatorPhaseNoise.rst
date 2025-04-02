=======================
Oscillator Phase Noise
=======================

.. inheritance-diagram:: hermespy.simulation.rf_chain.phase_noise.OscillatorPhaseNoise
   :parts: 1

Oscillator phase noise model according to :footcite:t:`2014:Khanzadi`,
modeling the phase noise of an oscillator as a function of the distance to the carrier frequency :math:`\Delta f` in frequency domain.
Phase noise is modeled as a superposition of three noise power spectral densities (PSDs)

.. math::

   S_{\phi}(\Delta f) = S_{\phi_0}(\Delta f) + S_{\phi_2}(\Delta f) + S_{\varphi_3}(\Delta f)

where

.. math::

   S_{\phi_0}(\Delta f) = K_0

denotes the white noise floor PSD of power :math:`K_0`,

.. math::

   S_{\phi_2}(\Delta f) = \frac{K_2}{f^2}

denotes the flicker noise PSD of power :math:`K_2` following a square law decay with distance to the carrier frequency :math:`\Delta f`
and

.. math::

   S_{\phi_3}(\Delta f) = \frac{K_3}{f^3}

denotes the flicker noise PSD of power :math:`K_3` following a cubic law decay with distance to the carrier frequency :math:`\Delta f`.
A starting point for the parameter values is given by :footcite:t:`2014:Khanzadi` as

.. math::

   K_0 &= -110~\mathrm{dB} = 10^{-110/10} \\
   K_2 &= 10 \\
   K_3 &= 10^4 \quad \text{.} \\

.. autoclass:: hermespy.simulation.rf_chain.phase_noise.OscillatorPhaseNoise

.. footbibliography::
