======
Matlab
======

Especially in engineering disciplines, `Matlab`_ by Mathworks is a popular scripting languange
for numerical simulations.
While Hermes is a Python tool, almost completely implemented in Python and C++, it is possible
to control modules and routines from Matlab's Python interface, esentially integrating Hermes'
full feature set into existing Matlab workflows.
Hermes requires Python 3.9, the respective Matlab interface requires a specific `Matlab Version`_.

Assuming Hermes has been properly :doc:`installed <installation>` on the host system,
before the execution of any python matlab routines, the virtual environment has to be selected:

.. code-block:: matlab

   pyenv('Version', '<path>', 'ExecutionMode', 'OutOfProcess');

Note that `path` has to point to the `python.exe` of your virtual environment.
So, for instance assuming Conda has been used to create Hermes' virtual Python environment, `<path>`
should look something like `C:\\Users\\<username>\\.conda\\envs\\<envname>\\python`, `<username>`
and `<envname>` being replaced by the respective directory names.

After environment activation, any Hermes module can be loaded, objects initialized and functions called.
As a minimal example, initializing a new simulation and adding a single virtual device can be achieved via

.. code-block:: matlab

   simulation_module = py.importlib.import_module('hermespy.simulation');
   simulation = simulation_module.Simulation();
   device = simulation.new_device();

Several complete examples are hosted within the official GitHub `repository`_.
They should provide a starting point for anyone interested in integrating Hermes functionalities
into their Matlab workflow.
Thanks to `Roberto Bomfin`_ for providing the source files.
Please note that Hermes does not officially support complete Matlab integration of all features.
Instead, this documentation is only served as a courtesy to users and developers.

.. _Matlab: https://mathworks.com/
.. _Matlab version: https://mathworks.com/support/requirements/python-compatibility.html
.. _repository: https://github.com/Barkhausen-Institut/hermespy/tree/main/_examples/settings/matlab
.. _Roberto Bomfin: https://www.linkedin.com/in/roberto-bomfin-b33a43144/