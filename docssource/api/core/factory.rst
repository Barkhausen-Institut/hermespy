=====================
Serialization Factory
=====================

.. inheritance-diagram:: hermespy.core.factory.Serializable hermespy.core.factory.SerializableEnum hermespy.core.factory.HDFSerializable hermespy.core.factory.Factory
   :parts: 1

This module implements the main interface for loading / dumping HermesPy configurations from / to `YAML`_ files.
Every mutable object that is expected to have its state represented as a text-section within configuration files
must inherit from the :class:`.Serializable` base class.

All :class:`.Serializable` classes within the `hermespy` namespace are detected automatically by the :class:`.Factory`
managing the serialization process.
As a result, dumping any :class:`.Serializable` object state to a `.yml` text file is as easy as

.. code-block:: python

   factory = Factory()
   factory.to_file("dump.yml", serializable)

and can be loaded again just as easily via

.. code-block::  python

        factory = Factory()
        serializable = factory.from_file("dump.yml")

from any context.


.. autoclass:: hermespy.core.factory.Serializable

.. autoclass:: hermespy.core.factory.SerializableEnum

.. autoclass:: hermespy.core.factory.HDFSerializable

.. autoclass:: hermespy.core.factory.Factory

.. autoclass:: hermespy.core.factory.SerializableType

.. autoclass:: hermespy.core.factory.SET

.. footbibliography::

.. _YAML: https://yaml.org/
