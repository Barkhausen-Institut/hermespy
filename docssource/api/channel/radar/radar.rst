===============
Radar Channels
===============

Hermes' radar channels can be used to simulate basic propagation characteristics
of sensing signals.
They can be thought of as stripped down raytracing channels which consider only
a single reflection in between the transmitting and receiving device,
with the reflection being generated by target objects specified by the user.

.. mermaid::
   :align: center

   classDiagram
       
      class RadarChannelBase {

         <<Abstract>>

         +realize()
         +propagate()
      }

      class SingleTargetRadarChannel {

         +propagate()
      }

      class MultiTargetRadarChannel {

         +propagate()
      }

      class RadarChannelRealization {

         <<Abstract>>

         +propagate()
      }

      class SingleTargetRadarChannelRealization {

         +propagate()
      }

      class MultiTargetRadarChannelRealization {

         +propagate()
      }

      class RadarPathRealization {

         <<Abstract>>

         +propagation_delay()
         +relative_velocity()
         +propagation_response()
      }

      class RadarInterferenceRealization {

         +propagation_delay()
         +relative_velocity()
         +propagation_response()
      }

      class RadarTargetRealization {

         <<Abstract>>

         +propagation_delay()
         +relative_velocity()
         +propagation_response()
      }

      class RadarTarget {

         <<Abstract>>

         +get_cross_section() : RadarCrossSectionModel
         +get_velocity()
         +get_forwards_transformation()
         +get_backwards_transformation()
      }

      class VirtualRadarTarget {

         +get_cross_section() : RadarCrossSectionModel
         +get_velocity()
         +get_forwards_transformation()
         +get_backwards_transformation()
      }

      class PhysicalRadarTarget {

         +RadarCrossSectionModel cross_section
         +Moveable moveable

         +get_cross_section() : RadarCrossSectionModel
         +get_velocity()
         +get_forwards_transformation()
         +get_backwards_transformation()
      }

      RadarChannelBase o-- RadarChannelRealization : realize()
      SingleTargetRadarChannel o-- SingleTargetRadarChannelRealization : realize()
      MultiTargetRadarChannel o-- MultiTargetRadarChannelRealization : realize()
      RadarChannelRealization *-- RadarPathRealization
      MultiTargetRadarChannel *-- RadarTarget

      SingleTargetRadarChannel --|> RadarChannelBase
      MultiTargetRadarChannel --|> RadarChannelBase

      SingleTargetRadarChannelRealization --|> RadarChannelRealization
      MultiTargetRadarChannelRealization --|> RadarChannelRealization
      RadarInterferenceRealization --|> RadarPathRealization
      RadarTargetRealization --|> RadarPathRealization
      PhysicalRadarTarget --|> RadarTarget
      VirtualRadarTarget --|> RadarTarget

      click RadarChannelBase href "channel.radar.RadarChannelBase.html"
      click SingleTargetRadarChannel href "channel.radar.SingleTargetRadarChannel.html"
      click MultiTargetRadarChannel href "channel.radar.MultiTargetRadarChannel.html"
      click RadarChannelRealization href "channel.radar.RadarChannelRealization.html"
      click SingleTargetRadarChannelRealization href "channel.radar.SingleTargetRadarChannelRealization.html"
      click MultiTargetRadarChannelRealization href "channel.radar.MultiTargetRadarChannelRealization.html"
      click RadarPathRealization href "channel.radar.RadarPathRealization.html"
      click RadarInterferenceRealization href "channel.radar.RadarInterferenceRealization.html"
      click RadarTargetRealization href "channel.radar.RadarTargetRealization.html"
      click RadarTarget href "channel.radar.RadarTarget.html"
      click VirtualRadarTarget href "channel.radar.VirtualRadarTarget.html"
      click PhysicalRadarTarget href "channel.radar.PhysicalRadarTarget.html"


There are currently two types of :class:`Radar Channels<hermespy.channel.radar.radar.RadarChannelBase>`,
namely the easy-to-use :class:`SingleTargetRadarChannel <hermespy.channel.radar.single.SingleTargetRadarChannel>`,
considering only a single reflector between its two linked devices,
and the more complex :class:`MultiTargetRadarChannel <hermespy.channel.radar.multi.MultiTargetRadarChannel>`,
which allows for the specification of multiple reflectors with individual :class:`Cross Sections<hermespy.channel.radar.multi.RadarCrossSectionModel>`,
that may represent :class:`Virtual Targets<hermespy.channel.radar.multi.VirtualRadarTarget>` or existing :class:`Physical Targets<hermespy.channel.radar.multi.PhysicalRadarTarget>`.

A radar channel linking two devices indexed by :math:`\alpha` and :math:`\beta` and located at cartesian coordinates :math:`\mathbf{p}_{\mathrm{Device}}^{(\alpha)}` and :math:`\mathbf{p}_{\mathrm{Device}}^{(\beta)}` while moving with a global velocity of
:math:`\mathbf{v}_{\mathrm{Device}}^{(\alpha)}` and :math:`\mathbf{v}_{\mathrm{Device}}^{(\beta)}` respectively,
is modeled by the impulse response

.. math::

   \mathbf{H}^{(\alpha,\beta)}(t, \tau) = \mathbf{H}_{\mathrm{LOS}}^{(\alpha, \beta)}(t) \delta(\tau - \tau_{\mathrm{Device}}^{(\alpha,\beta)}) + \sum_{\ell=1}^{L} \mathbf{H}_{\mathrm{Target}}^{(\alpha, \beta, \ell)}(t) \delta(\tau - \tau_{\mathrm{Target}}^{(\alpha,\beta,\ell)})

considering a line-of-sight (LOS) interference component between both devices
and a sum of :math:`L` first-order reflection components generated by the :math:`L` targets.
The LOS component

.. math::

   \mathbf{H}_{\mathrm{LOS}}^{(\alpha, \beta)}(t) = \frac{ c_0 }{ 4 \pi f_{\mathrm{c}}^{(\alpha)} d_{\mathrm{Device}}^{(\alpha,\beta)}  } \mathbf{A}_{\mathrm{Device}}^{(\alpha,\beta)} \exp\left( 2\mathrm{j}\pi f_{\mathrm{c}}^{(\alpha)} (\tau_{\mathrm{Device}}^{(\alpha,\beta)} + \frac{ \overline{v}_{\mathrm{Device}}^{(\alpha,\beta)}}{ c_0 } t ) \right)

depends on the distance and resulting time-of-flight delay between the devices

.. math::

   d_{\mathrm{Device}}^{(\alpha,\beta)} &= \| \mathbf{p}_{\mathrm{Device}}^{(\alpha)} - \mathbf{p}_{\mathrm{Device}}^{(\beta)} \|_2 \\
   \tau_{\mathrm{Device}}^{(\alpha,\beta)} &= \frac{ d_{\mathrm{Device}}^{(\alpha,\beta)} }{ c_0 } \\

as well as the relative velocity between the devices

.. math::

   \overline{v}_{\mathrm{Device}}^{(\alpha,\beta)} = \frac{ (\mathbf{v}_{\mathrm{Device}}^{(\alpha)} - \mathbf{v}_{\mathrm{Device}}^{(\beta)})^\mathsf{T} (\mathbf{p}_{\mathrm{Device}}^{(\alpha)} - \mathbf{p}_{\mathrm{Device}}^{(\beta)} ) }{ d_{\mathrm{Device}}^{(\alpha,\beta)} } \ \text{.}

Note that  if a mono-static radar configuration is considered, i.e. :math:`\alpha = \beta`,
the LOS component is not considered, since it is equivalent to :doc:`Leakage</api/simulation/isolation>` between transmit and receive chains.
The reflection components generated by the :math:`L` targets are modeled by

.. math::

   \mathbf{H}_{\mathrm{Target}}^{(\alpha, \beta, \ell)}(t) = \frac{ c_0 \sigma_{\ell}^{\frac{1}{2}} }{ (4 \pi)^{\frac{3}{2}} f_{\mathrm{c}}^{(\alpha)} d_{\mathrm{Target}}^{(\alpha,\ell)} d_{\mathrm{Target}}^{(\beta,\ell)} }
   \exp\left( 2\mathrm{j}\pi f_{\mathrm{c}}^{(\alpha)} (\tau_{\mathrm{Target}}^{(\alpha,\beta,\ell)} + \frac{ \overline{v}_{\mathrm{Target}}^{(\alpha,\ell)} + \overline{v}_{\mathrm{Target}}^{(\beta,\ell)}}{ c_0 } t ) +\mathrm{j} \phi_{\mathrm{Target}}^{(\ell)} \right)

where the distance and resulting time-of-flight delay between the device and the :math:`\ell`-th target are given by

.. math::

   d_{\mathrm{Target}}^{(\alpha,\ell)} &= \| \mathbf{p}_{\mathrm{Device}}^{(\alpha)} - \mathbf{p}_{\mathrm{Target}}^{(\ell)} \|_2 \\
   \tau_{\mathrm{Target}}^{(\alpha,\beta,\ell)} &= \frac{ d_{\mathrm{Target}}^{(\alpha,\ell)} + d_{\mathrm{Target}}^{(\beta,\ell)} }{ c_0 } \ \text{.}

The doppler shift perceived by the receiving device depends on the velocities on both devices and the target,
which results in an overall relative velocity of

.. math::

   \overline{v}_{\mathrm{Target}}^{(\alpha,\ell)} = \frac{ (\mathbf{v}_{\mathrm{Target}}^{(\ell)} - \mathbf{v}_{\mathrm{Device}}^{(\alpha)})^\mathsf{T} (\mathbf{p}_{\mathrm{Target}}^{(\ell)} - \mathbf{p}_{\mathrm{Device}}^{(\alpha)} ) }{ d_{\mathrm{Target}}^{(\alpha,\ell)} } \ \text{.}

.. autoclass:: hermespy.channel.radar.radar.RadarChannelBase

.. autoclass:: hermespy.channel.radar.radar.RadarChannelRealization

.. autoclass:: hermespy.channel.radar.radar.RCRT

.. autoclass:: hermespy.channel.radar.radar.RadarChannelSample

.. autoclass:: hermespy.channel.radar.radar.RCST

.. autoclass:: hermespy.channel.radar.radar.RadarPath

.. autoclass:: hermespy.channel.radar.radar.RadarTargetPath

.. autoclass:: hermespy.channel.radar.radar.RadarInterferencePath

.. toctree::
   :glob:
   :hidden:

   single
   multi
