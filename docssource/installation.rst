Installation
===============

There are two supported ways to install HermesPy on your system.
Users may either pull pre-built wheels from the official python package index `PyPi`_ or
or clone the source code from the `GitHub`_ and build the package from scratch.

Both methods will be described in the following sections.

Install from PyPi
-----------------

Pulling HermesPy from `PyPi`_ is the recommended installation method for end-users who do not
intend to extend the HermesPy source code.

HermesPy is registered as an official `package`_ within the python package index.
Binaries for Windows, most Linux distributions and MacOS, built from the newest release version
of HermesPy, are distributed via the index, enabling easy installations for most operating systems.

Install the package via

##### Windows users:
```commandline
conda create -n <envname> python=3.9
conda activate <envname>
conda install pip
pip install hermespy
```

##### Linux users
```commandline
python -m venv env
. env/bin/activate
pip install hermespy
```

Install from Source
-------------------

This is the recommended method for developers.
You can build the package from scratch at any system by cloning the repository source via
```commandline
git clone <this-repo>
cd hermespy/
```
Make sure to have [Git LFS](https://git-lfs.github.com/) installed.
#### Windows users
1. Create a new virtual environment using conda or any environment manager of your choice and install the default
   python package manager pip
   ```commandline
   conda create -n <envname> python=3.9
   conda activate <envname>
   conda install pip
   ```
2. Install HermesPy as well as its dependencies using pip
   ````commandline
   pip install -r requirements.txt
   pip install -e .
   ````

#### Linux users
   1. Make sure the `python` symlink is linked to python3.9
   2. Create a new virtual environment using venv or any environment manager of your choice and install the default
      python package manager pip
      ```commandline
      python -m venv env
      . env/bin/activate
      ```
      3. Install HermesPy as well as its dependencies using pip
      ```commandline
      pip install -r requirements.txt
      pip install -e .
      ```

[Quadriga channel model v0.2.2](https://quadriga-channel-model.de/) is supported by HermesPy.
For it to be used, some preliminary steps need to be taken.
It can be run with either Octave or matlab. For **octave**, under Windows, you need to set the environment variable that tells python where to find octave-cli executable by calling

```commandline
setx PATH "%PATH%;<path-to-octave-cli>
```

and install `oct2py` via `pip install oct2py` (Ubuntu sets the environment variable automatically).

If you want to run **matlab**, you must use the `matlab.engine` package provided by Matlab.
Refer to [this](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) link.

.. _PyPi: https://pypi.org/
.. _GitHub: https://github.com/Barkhausen-Institut/hermespy
.. _package: https://pypi.org/project/hermespy/
