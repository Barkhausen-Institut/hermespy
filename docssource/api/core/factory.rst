==============
Serialization
==============

.. inheritance-diagram:: hermespy.core.factory.Serializable hermespy.core.factory.SerializableEnum hermespy.core.factory.Factory hermespy.core.factory.ProcessBase hermespy.core.factory.SerializationProcess hermespy.core.factory.DeserializationProcess hermespy.core.factory.HDFSerializationProcess hermespy.core.factory.HDFDeserializationProcess
   :parts: 1

This module implements the main interface for saving and loading HermesPy's runtime objects to and from disk.


All :class:`.Serializable` classes within the `hermespy` namespace are detected automatically by the :class:`.Factory`
managing the serialization process.
As a result, dumping any :class:`.Serializable` object state to a file is as simple as

.. code-block:: python

   # Initialize a new serialization process
   factory = Factory()
   process = factory.serialize('file')

   # Serialize the object
   process.serialize_object(serializable_object, 'identifier')

   # Close the serialization process
   process.finalize()


and can be loaded again just as easily via

.. code-block::  python

   # Initialize a new deserialization process
   factory = Factory()
   process = factory.deserialize('file')

   # Deserialize the object
   deserialized_object = process.deserialize_object('identifier', SerializableType)

   # Close the deserialization process
   process.finalize()


from any context.
By default, the objects will be serialized in the `HDF`_ format,
which is currently the only supported format for serialization.
Future versions of HermesPy may support additional serialization formats such as 
Matlab or JSON.

.. autoclass:: hermespy.core.factory.Serializable

.. autoclass:: hermespy.core.factory.SerializableEnum

.. autoclass:: hermespy.core.factory.Factory

.. autoclass:: hermespy.core.factory.ProcessBase

.. autoclass:: hermespy.core.factory.SerializationProcess

.. autoclass:: hermespy.core.factory.DeserializationProcess

.. autoclass:: hermespy.core.factory.HDFSerializationProcess

.. autoclass:: hermespy.core.factory.HDFDeserializationProcess

.. autoclass:: hermespy.core.factory.SerializableType

.. autoclass:: hermespy.core.factory.SET

.. footbibliography::

.. _HDF: https://www.h5py.org/
