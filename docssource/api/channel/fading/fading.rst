=================
Multipath Fading
=================

This module provides several implementations for statistical time-variant channel models.
A single multipath propagation path is modeled as a Rician fading distribution
by a sum-of-sinusoids approach

.. math::

   h_{\ell}(t) = 
      \sqrt{\frac{K_{\ell}}{1 + K_{\ell}}} \mathrm{e}^{\mathrm{j} t \omega_{\ell} \cos(\theta_{\ell,0}) + \mathrm{j} \phi_{\ell,0} }
      + \sqrt{\frac{1}{N(1 + K_{\ell})}} \sum_{n=1}^{N} \mathrm{e}^{\mathrm{j} t \omega_{\ell} \cos\left( \frac{2\pi n + \theta_{\ell,n}}{N} \right) + \mathrm{j} \phi_{\ell,n}}

as proposed by :footcite:t:`2006:xiao`.
Each propagation path is composed of a specular component and a diffuse component,
with rice factor :math:`K_{\ell}` balancing the power distribution between the both components.
The more sinusoids :math:`N` are summed to model the fading, the more accurate the model is, however, :footcite:t:`2006:xiao` indicate that :math:`N = 8` is a good starting point
for balancing accuracy with computational complexity.
The overall path has a doppler shift :math:`\omega_{\ell}` that is balanced by the angles :math:`\theta_{\ell,0}` and :math:`\theta_{\ell,n}` for line of sight and diffuse components, respectively,
with a random phase :math:`\phi_{\ell,0}` and :math:`\phi_{\ell,n}`.
Note that for :math:`K_{\ell} = 0`, i.e. a non line of sight fading consisting of diffuse components only, this approximates a Rayleigh distribution.
This model can is extended by :math:`L` spatial delay taps, meaning there are multiple spatial propagation paths resulting in a delay spread of the transmitted signal at the receiver,
so that the overall channel is the sum of all paths 

.. math::

   \mathbf{H}(t,\tau) = \mathbf{A}^{(0)} \sum_{\ell=1}^{L} g_{\ell} h_{\ell}(t) \delta(\tau - \tau_{\ell}) \mathbf{A}^{(1)} \ \text{,}

with :math:`g_{\ell}` being the gain factor of the :math:`\ell`-th path, :math:`\tau_{\ell}` the delay of the :math:`\ell`-th path, and :math:`\mathbf{A}^{(0)}` and :math:`\mathbf{A}^{(1)}` being the antenna correlation matrices of the transmitter and receiver, respectively.
For MIMO configurations with :class:`Devices<hermespy.simulation.simulated_device.SimulatedDevice>` featuring multiple transmit- or receive-antennas,
custom antenna correlations :math:`A^{(0)}` and :math:`A^{(1)}` can be specified for both linked devices, respectively, as proposed by :footcite:t:`2002:yu`.
However, these antenna correlations do not model antenna array responses from spatial wave impingements.
Instead, they are purely statistical approximations.

.. mermaid::

   classDiagram
       
      class MultipathFadingChannel {

         +realize()
         +propagate()
      }

      class MultipathFadingRealization {

         +propagate()
      }

      class PathRealization {

         +propagate()
      }


      class MultipathFading5GTDL {

         +realize()
         +propagate()
      }


      class MultipathFadingCost259 {

         +realize()
         +propagate()
      }


      class MultipathFadingExponential {

         +realize()
         +propagate()
      }


      MultipathFadingChannel --o MultipathFadingRealization : realize()
      PathRealization --* MultipathFadingRealization
      MultipathFading5GTDL --|> MultipathFadingChannel
      MultipathFadingCost259 --|> MultipathFadingChannel
      MultipathFadingExponential --|> MultipathFadingChannel

      click MultipathFadingChannel href "channel.multipath_fading_channel.MultipathFadingChannel.html"
      click MultipathFadingRealization href "channel.multipath_fading_channel.MultipathFadingRealization.html"
      click PathRealization href "channel.multipath_fading_channel.PathRealization.html"
      click MultipathFading5GTDL href "channel.multipath_fading_templates.MultipathFading5GTDL.html"
      click MultipathFadingCost259 href "channel.multipath_fading_templates.MultipathFadingCost259.html"
      click MultipathFadingExponential href "channel.multipath_fading_templates.MultipathFadingExponential.html"


The base equations are implemented in the :class:`hermespy.channel.fading.fading.MultipathFadingChannel` and its respective
:class:`hermespy.channel.fading.fading.MultipathFadingRealizaiton` / :class:`hermespy.channel.fading.fading.MultipathFadingSample`.
Users may directly use the :class:`hermespy.channel.fading.fading.MultipathFadingChannel` with their own parameter sets.
More conveniently, several standard parameterizations such as :class:`hermespy.channel.fading.tdl.TDL`
:class:`hermespy.channel.fading.cost259.Cost259` and :class:`hermespy.channel.fading.exponential.Exponential`
are provided.

.. toctree::
   :glob:
   :hidden:

   cost259
   exponential
   tdl
   fading.*

.. footbibliography::
