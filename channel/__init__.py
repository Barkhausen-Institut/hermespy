from __future__ import annotations
from .channel import Channel

__all__ = [Channel]

# Register serializable classes to YAML factory
import simulator_core as core
core.SerializableClasses.add(Channel)
