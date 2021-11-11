# HermesPy
HermesPy (Heterogeneous Radio Mobile Simulator - Python) is a semi-static link-level simulator based on time-driven mechanisms.

It provides a framework for the link-level simulation of a multi-RAT wireless link, consisting of
multiple transmit and receive modems, which may operate at different carrier frequencies. Besides
simulating individual transmission links, HermesPy allows the analysis of both co-channel and
adjacent-channel interference among different communication systems.

You can find an introductory video here: https://www.barkhauseninstitut.org/en/results/hermespy

## Features

The current release version 0.2.0 serves as a platform for joint development.
Its main focus is to provide a software architecture that can be easily extended.
Detailed core features as well as a release plan can be found in the [features](FEATURES.md) section.

## Installation
There are two supported ways to install HermesPy on your system:

#### From PyPI
This is the recommended method for end-users.
Hermespy is registered as an [official package](https://pypi.org/project/hermespy/) in PyPI.
We intend to directly serve prebuilt binaries for Windows, most Linux distributions and MacOS.
Install the package via

##### Windows users:
```commandline
conda create -n <envname> python=3.9
conda activate <envname>
conda install pip
pip install hermespy
```

##### Linux users
````commandline
python -m venv env
. env/bin/activate
pip install hermespy
````

### From Source
This is the recommended method for developers.
You can build the package from scratch at any system by cloning the repository source via
````commandline
git clone <this-repo>
cd hermespy/
````
#### Windows users
1. Create a new virtual environment using conda or any environment manager of your choice and install the default
   python package manager pip
   ````commandline
   conda create -n <envname> python=3.9
   conda activate <envname>
   conda install pip
   ````
2. Install HermesPy as well as its dependencies using pip
   ````commandline
   pip install -r requirements.txt
   pip install -e .
   ````

#### Linux users
   1. Make sure the `python` symlink is linked to python3.9
   2. Create a new virtual environment using venv or any environment manager of your choice and install the default
      python package manager pip
      ````commandline
      python -m venv env
      . env/bin/activate
      ````
      3. Install HermesPy as well as its dependencies using pip
      ````commandline
      pip install -r requirements.txt
      pip install -e .
      ````

[Quadriga channel model v0.2.0](https://quadriga-channel-model.de/) is supported by HermesPy.
For it to be used, some preliminary steps need to be taken.
It can be run with either Octave or matlab. For **octave**, under Windows, you need to set the environment variable that tells python where to find octave-cli executable by calling

```commandline
setx PATH "%PATH%;<path-to-octave-cli>
```

and install `oct2py` via `pip install oct2py` (Ubuntu sets the environment variable automatically).

If you want to run **matlab**, you must use the `matlab.engine` package provided by Matlab.
Refer to [this](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) link.

## HermesPy as a Command Line Tool

1. Activate your virtual environment:
   1. **Windows**: `conda activate <envname>`
   2. **ubuntu**: `. env/bin/activate` 
2. The simulator can be called by `hermes <param> <output_dir>`.

The main execution is `hermes -p <settings-directory> -o <results-directory>`.
Both command line parameters are optional arguments.
`<settings-directory>` contains settings files for the simulation and `<results-directory>` tells the simulator where to put the result files.
Both are relative paths and have default values namely `_settings` and `results`.
All parameter values in the settings files are in SI units. Please refer to our documentation (see below).

Some thoroughly commented example configurations highlighting currently available HermesPy features may be found in the **_examples** folder.
Check the readme files for further information.

### Quadriga

The Quadriga channel model engine will be called by HermesPy for each simulation drop, as soon as at least on channel within a scenario configuration is a Quadriga-type channel.
Check the example provided under `_examples/_quadriga` for a reference configuration.

**Important note**: If you decide to run HermesPy in combination with Quadriga, make sure either Matlab or Octave,
as well as their respective python bindings [oct2py](https://pypi.org/project/oct2py/) or [matlab.engine](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) are installed.
At least one interpreter must be present, otherwise HermesPy will abort during execution.
Should both interpreters be installed, HermesPy will select Matlab by default.

## HermesPy as a Python Library

In addition to using HermesPy as a command-line tool configured by *.yml* configuration files, it may be used in your own python projects as a drop-in library for wireless transmission link simulation.
Make sure you install HermesPy within the same virtual environment as the project you want to use it in.

The library follows a handle-based approach, giving users an easy and quick way to configure a simulation scenario.
For example, generating a 1ms Frequency-Shift-Keying waveform transmitted by a modem featuring one antenna can be done via
```python
from hermespy import Scenario, Transmitter, WaveformGeneratorChirpFsk

scenario = Scenario()

transmitter = Transmitter()
scenario.add_transmitter(transmitter)

transmitter.waveform_generator = WaveformGeneratorChirpFsk()
waveform = transmitter.send(1e-3)
```
The library interface to HermesPy offers complete freedom to re-configure scenarios during simulation runtime and
perform Monte-Carlo simulations over arbitrary sets of configuration parameters. Refer to the [documentation](https://barkhausen-institut.github.io/hermespy/index.html)
for further information.

## Documentation

Documentation can be found [here](https://barkhausen-institut.github.io/hermespy/index.html). It is auto-generated by the GitHub action.
Quadriga documentation can be found in **hermes/docssource**.

## Known Limitations

- No parallelization of the simulation
- General performance improvements required

## Contributors

* [Andre Noll Barreto](https://gitlab.com/anollba)
* [Tobias Kronauer](https://github.com/tokr-bit)
* [Jan Adler](https://github.com/adlerjan)

## Copyright
Copyright (C) 2021 Barkhausen Institut gGmbH

