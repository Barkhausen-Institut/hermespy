from .scenario import Scenario

__all__ = ['Scenario']

# Register serializable classes to YAML factory
import simulator_core as core
core.SerializableClasses.add(Scenario)
