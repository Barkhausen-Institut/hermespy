Installation
===============

There are two supported ways to install HermesPy on your system.
Users may either pull pre-built wheels from the official python package index `PyPi`_ or
or clone the source code from `GitHub`_ and build the package from scratch.

Both methods will be described in the following sections.

Install from PyPi
-----------------

Pulling HermesPy from `PyPi`_ is the recommended installation method for end-users who do not
intend to extend the HermesPy source code.

HermesPy is registered as an official `package`_ within the python package index.
Binaries for Windows, most Linux distributions and MacOS, built from the newest release version
of HermesPy, are distributed via the index, enabling easy installations for most operating systems.

Install the package by executing the following commands within a terminal:

.. tabs::

   .. code-tab:: batch Windows

      conda create -n <envname> python=3.9
      conda activate <envname>
      conda install pip
      pip install hermespy

   .. code-tab:: bash Linux

      python -m venv env
      . env/bin/activate
      pip install hermespy

Executing these statements sequentially results in the following actions:

#. Creation of a new virtual environment titled `<envname>`
#. Activate the newly created environment
#. Install the HermesPy wheel from PyPi within the environment

Note that if you plan on utilizing HermesPy within an already existing Python environment,
you may omit step one and replace `<envname>` by the title of the existing environment.

Install from Source
-------------------

Cloning the HermesPy source code and manually building / installing its package is the recommended way
for developers who plan on extending the HermesPy source code.
Additionally, it can also be applied by users who, for any reason, are unable to install HermesPy from
the index.

Before cloning, make sure to have the `LFS`_ extension to `Git`_ installed.
Using the `Git`_ command line interface,
the HermesPy source code can be copied to any system by executing

.. code-block:: bash

   git clone <this-repo>
   cd hermespy/

within a terminal.


Some submodules of HermesPy are provided as C++ implementations with Python bindings for improved performance.
Therefore, building the package from source requires your system to have a
build chain detectable by `CMake`_ installed and configured.
For Windows users, we recommend downloading and installing either the `Visual Studio Build Tools`_
or `MinGW`_ as a free open-source alternative.


Build and install the package contained within the repository by executing the following commands within a terminal:

.. tabs::

   .. code-tab:: batch Windows

      conda create -n <envname> python=3.9
      conda activate <envname>
      conda install pip
      pip install .

   .. code-tab:: bash Linux

      python -m venv env
      . env/bin/activate
      pip install .

Executing these statements sequentially results in the following actions:

#. Creation of a new virtual environment titled `<envname>`
#. Activate the newly created environment
#. Install the HermesPy wheel from source within the environment

Note that if you plan on utilizing HermesPy within an already existing Python environment,
you may omit step one and replace `<envname>` by the title of the existing environment.

**If you plan to alter the source code in any way, we recommend appending the editable flag**

.. code-block:: bash

   pip install -e .

**during installation.**

Install Quadriga
----------------

In addition to its native channel models, HermesPy supports the `Quadriga`_ channel model as an external
dependency.
For it to be used, some preliminary steps need to be taken.
`Quadriga`_ is based on `Matlab`_ and can be executed by either the `Matlab`_ interpreter or its open-source
equivalent `Octave`_.

In order to execute the `Matlab`_ interpreter the `matlab.engine`_ package provided by `Matlab`_ needs to be installed
manually.

In order to execute the `Octave`_ interpreter the additional `oct2py`_ package needs to be installed
(`pip install oct2py`).
Under Windows, an extension of the `PATH` variable may be required for `oct2py`_ to be able to locate the octave
command line interface:

.. code-block:: bash

   setx PATH "%PATH%;<path-to-octave-cli>"

When installing HermesPy from the distributed `package`_, the Quadriga source code needs to be installed manually.
Download the latest version of `Quadriga`_ and extract the zip archive in a location of your choice.
Afterwards, set the environment variable `HERMES_QUADRIGA` to point to the `quadriga_src` directory.
This will point Hermes to search for the Quadriga files within the specified location during simulation runtime.

.. _PyPi: https://pypi.org/
.. _GitHub: https://github.com/Barkhausen-Institut/hermespy
.. _package: https://pypi.org/project/hermespy/
.. _Git: https://git-scm.com/
.. _LFS: https://git-lfs.github.com/
.. _Quadriga: https://quadriga-channel-model.de/
.. _Matlab: https://www.mathworks.com/products/matlab.html
.. _Octave: https://www.gnu.org/software/octave/index
.. _matlab.engine: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
.. _oct2py: https://pypi.org/project/oct2py/
.. _CMake: https://cmake.org/
.. _Visual Studio Build Tools: https://visualstudio.microsoft.com/de/downloads/#build-tools-for-visual-studio-2022
.. _MinGW: https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/mingw-w64-install.exe/download