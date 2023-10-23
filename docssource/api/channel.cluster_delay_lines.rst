==============================
3GPP Cluster Delay Line Models
==============================

Within this module, HermesPy implements the 3GPP standard for cluster delay line models
as defined in the :footcite:t:`3GPP:TR38901`.

For a link between two devices :math:`\alpha` and :math:`\beta` featuring :math:`N^{(\alpha)}` and :math:`N^{(\beta)}` antennas, respectively,
the model assumes that the channel impulse response is composed of a sum of propagation paths between :math:`C` clusters of scatterers perceived by both devices,
with each cluster containing :math:`L` individual scatterers, therefore resulting in :math:`C \cdot L` propagation paths between each antenna pair
and :math:`C \cdot L \cdot N^{(\alpha)} \cdot N^{(\beta)}` propagation paths in total.

The impulse response of each propagation path within the cluster delay model

.. math::

   h^{(\alpha, \beta)}(t, \tau) = \sqrt{\frac{1}{1 + K}} h^{(\alpha, \beta)}_{\mathrm{NLOS}}(t, \tau) + \sqrt{\frac{K}{1 + K}} h^{(\alpha, \beta)}_{\mathrm{LOS}}(t, \tau)

is a sum of a non-line-of-sight (NLOS) and a line-of-sight (LOS) component, balanced by the Ricean :math:`K`-factor.
Both the NLOS and LOS components 

..
   h^{(\alpha, \beta)}_{\mathrm{LOS}}(t, \tau) =
   \begin{split}
   & \mathbf{F}^{(\beta)}(\theta_{\mathrm{LOS,ZOA}}, \phi_{\mathrm{LOS,AOA}})^{\mathsf{T}}
   \begin{bmatrix}
      1 & 0 \\
      0 & -1 \\
   \end{bmatrix}
   \mathbf{F}^{(\alpha)}(\theta_{\mathrm{LOS,ZOD}}, \phi_{\mathrm{LOS,AOD}}) \\
   & \cdot \exp\left(
      - \mathrm{j}2\pi\frac{d_{\mathrm{3D}}}{\lambda}
      + \mathrm{j}2\pi\frac{r^{(\beta,\alpha) + \wideline{v}t}{\lambda}
      + \mathrm{j}2\pi\frac{r^{(\beta,\alpha)} t}{\lambda}
      \right)
   \end{split}

are functions of the two :doc:`Antennas'<core.antennas>` polarization :meth:`characteristics<hermespy.core.antennas.Antenna.local_characteristics>`
:math:`\mathbf{F}^{(a)}(\theta, \phi)`
towards angle-of-arrival :math:`\theta_{\mathrm{ZOA}}, \phi_{\mathrm{AOA}}`
and angle-of-departure :math:`\theta_{\mathrm{ZOD}}, \phi_{\mathrm{AOD}}`.
For a comprehensive description of all the parameters involved, please refer to the standard document.

The following standard parameterizations are currently provided by HermesPy:

=================   ==============================================================================   =================================================================================  ==================================================================================
Model               Line Of Sight                                                                    No Line Of Sight                                                                   Outside To Inside
=================   ==============================================================================   =================================================================================  ==================================================================================
Indoor Factory      :doc:`channel.cluster_delay_line_indoor_factory.IndoorFactoryLineOfSight`        :doc:`channel.cluster_delay_line_indoor_factory.IndoorFactoryNoLineOfSight`      
Indoor Office       :doc:`channel.cluster_delay_line_indoor_office.IndoorOfficeLineOfSight`          :doc:`channel.cluster_delay_line_indoor_office.IndoorOfficeNoLineOfSight`        
Rural Macrocells    :doc:`channel.cluster_delay_line_rural_macrocells.RuralMacrocellsLineOfSight`    :doc:`channel.cluster_delay_line_rural_macrocells.RuralMacrocellsNoLineOfSight`    :doc:`channel.cluster_delay_line_rural_macrocells.RuralMacrocellsOutsideToInside`
Street Canyon       :doc:`channel.cluster_delay_line_street_canyon.StreetCanyonLineOfSight`          :doc:`channel.cluster_delay_line_street_canyon.StreetCanyonNoLineOfSight`          :doc:`channel.cluster_delay_line_street_canyon.StreetCanyonOutsideToInside`      
Urban Macrocells    :doc:`channel.cluster_delay_line_urban_macrocells.UrbanMacrocellsLineOfSight`    :doc:`channel.cluster_delay_line_urban_macrocells.UrbanMacrocellsNoLineOfSight`    :doc:`channel.cluster_delay_line_urban_macrocells.UrbanMacrocellsOutsideToInside`
=================   ==============================================================================   =================================================================================  ==================================================================================

These preset standard parameterizations distinguish between line-of-sight, no line-of-sight and, in some cases, outside-to-inside propagation conditions.
For custom parameterizations, :doc:`channel.cluster_delay_lines.ClusterDelayLine` offers an extended interface to define each parameter of the cluster delay line model individually.

.. mermaid::

     classDiagram
 
         class Channel {
 
             <<Abstract>>
 
             _realize()
         }

         Channel --o ChannelRealization
         ClusterDelayLineRealization --|> ChannelRealization
         ClusterDelayLineBase --|> Channel
         ClusterDelayLineBase --o ClusterDelayLineRealization
         IndoorFactoryBase --|> ClusterDelayLineBase
         UrbanMicroCellsNoLineOfSight --|> ClusterDelayLineBase
         ClusterDelayLine --|> ClusterDelayLineBase
         ClusterDelayLineIndoorFactory --|> IndoorFactoryBase
         ClusterDelayLineIndoorOffice --|> ClusterDelayLineBase
         ClusterDelayLineRuralMacrocells --|> ClusterDelayLineBase
         ClusterDelayLineStreetCanyon --|> ClusterDelayLineBase
         ClusterDelayLineUrbanMacrocells --|> ClusterDelayLineBase
         StreetCanyonNoLineOfSight --|> UrbanMicroCellsNoLineOfSight
         StreetCanyonOutsideToInside --> UrbanMicroCellsNoLineOfSight

.. toctree::
   :hidden:
   :glob:
    
   channel.cluster_delay_line_*
   channel.cluster_delay_lines.ClusterDelayLineBase
   channel.cluster_delay_lines.ClusterDelayLine
   channel.cluster_delay_lines.ClusterDelayLineRealization
   channel.cluster_delay_lines.DelayNormalization


.. footbibliography::
