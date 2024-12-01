===========
Beamforming
===========

The beamforming module of HermesPy provides routines for spatial precoding of MIMO antenna streams.

.. autoclasstree:: hermespy.beamforming
   :strict:
   :namespace: hermespy

Beamforming during transmission allows to steer the power of a signal emitted by an antenna array towards
a desired direction, within the package this direction is referred to as focus point.
During signal reception inverse beamforming can be used as well in order to focus the received signal power towards a focus point.
By focusing towards a dictionary of angles of interest during signal reception, sensing algorithms ma create a spatial image of an array's environment.
Within this package, this process is referred to as :func:`probing <hermespy.beamforming.beamformer.ReceiveBeamformer.probe>`.

.. toctree::
   :maxdepth: 0
   :titlesonly:
   
   beamforming.beamformer
   beamforming.conventional
   beamforming.capon
   beamforming.nullsteeringbeamformer