# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Callable, Generic, TypeVar

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


_RT = TypeVar("_RT")
"""Type of operation result."""


class Hook(Generic[_RT]):
    """Hook for a callback to be called after a specific DSP layer was called."""

    __callback: Callable[[_RT], None]

    def __init__(self, hookable: Hookable[_RT], callback: Callable[[_RT], None]) -> None:
        """
        Args:

            hookable (Hookable[_RT]):
                DSP layer to be hooked into by the represented callback.

            callback (Callable[[_RT], None]):
                Function to called after processing the DSP layer.
                The DSP layer's output is passed as the only argument.
        """

        # Initialize attributes
        self.__hookable = hookable
        self.__callback = callback

    def __call__(self, output: _RT) -> None:
        """Call the callback function with the output of the DSP layer.

        Args:

            output (_RT):
                Output of the DSP layer.
        """

        self.__callback(output)

    def remove(self) -> None:
        """Remove the callback from the DSP layer."""

        self.__hookable.remove_hook(self)


class Hookable(Generic[_RT]):
    """Base class of DSP layers that can be hooked into by callbacks."""

    __hooks: set[Hook[_RT]]  # Set of unique hooks representing callbacks

    def __init__(self) -> None:
        # Initialize attributes
        self.__hooks = set()

    def add_hook(self, hook: Hook[_RT]) -> None:
        """Add a callback to be called after processing the DSP layer.

        Args:

            hook (Hook[_RT]):
                Hook to be added.
        """

        self.__hooks.add(hook)

    def add_callback(self, callback: Callable[[_RT], None]) -> Hook[_RT]:
        """Add a callback to be called after processing the DSP layer.

        Instantiates a new :class:`Hook` object representing the provided `callback`
        function.
        Note that each :class:`Hook` instance should notify the :class:`Hookable`
        by calling its :meth:`Hook.remove` method once the represented callback is no longer required.

        Args:

            callback (Callable[[_RT], None]):
                Function to called after processing the DSP layer.
                The DSP layer's output is passed as the only argument.

        Returns: The added callback hook.
        """

        hook = Hook(self, callback)
        self.add_hook(hook)
        return hook

    def remove_hook(self, hook: Hook[_RT]) -> None:
        """Remove a callback hook from this DSP layer.

        Args:

            hook (Hook[_RT]):
                Hook to be removed.
        """

        self.__hooks.discard(hook)

    def notify(self, output: _RT) -> None:
        """Notify all registered callbacks of the DSP layer.

        Args:

            output (_RT):
                Output of the DSP layer.
        """

        for hook in self.__hooks:
            hook(output)
