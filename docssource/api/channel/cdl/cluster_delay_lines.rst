==============================
3GPP Cluster Delay Line Models
==============================

.. inheritance-diagram:: hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineBase hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineRealization hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineSample
   :parts: 1

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

are functions of the two :doc:`Antennas'</api/core/antennas>` polarization :meth:`characteristics<hermespy.core.antennas.Antenna.local_characteristics>`
:math:`\mathbf{F}^{(a)}(\theta, \phi)`
towards angle-of-arrival :math:`\theta_{\mathrm{ZOA}}, \phi_{\mathrm{AOA}}`
and angle-of-departure :math:`\theta_{\mathrm{ZOD}}, \phi_{\mathrm{AOD}}`.
For a comprehensive description of all the parameters involved, please refer to the standard document.

The following standard parameterizations are currently provided by HermesPy:

======================= ===============================   
Model                   Description
======================= ===============================  
:doc:`cdl`              Spatially invariant CDL models.
:doc:`indoor_factory`   Model of a factory hall.    
:doc:`indoor_office`    Model of an office building.   
:doc:`rural_macrocells` Model of a rural area.    
:doc:`urban_macrocells` Model of an urban area.   
:doc:`urban_microcells` Model of a street canyon.  
======================= ===============================   

These preset standard parameterizations distinguish between line-of-sight, no line-of-sight and, in some cases, outside-to-inside propagation conditions.

.. toctree::
   :hidden:
    
   cdl
   indoor_factory
   indoor_office
   rural_macrocells
   urban_macrocells
   urban_microcells

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineBase

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineRealization

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineSample

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines.LargeScaleState

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines.CDLRT

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines.LSST

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines._PowerDelayVisualization

.. autoclass:: hermespy.channel.cdl.cluster_delay_lines._AngleVisualization 

.. footbibliography::
