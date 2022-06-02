#!/usr/bin/env python

"""
=============================
Hermes Command Line Interface
=============================

Within this module the primary entry point to the HermesPy simulation routine is defined as
command line interface (CLI).
In order to conveniently launch HermesPy from any command line,
make sure that the virtual Python environment you installed HermesPy in is activated by executing

.. tabs::

   .. code-tab:: powershell Conda

      conda activate <envname>

   .. code-tab:: batch VEnv

      <envname>\\Scripts\\activate.bat

within a terminal.
Usually activated environments are indicated by a `(<envname>)` in front of your terminal command line.
Afterwards, HermesPy can be launched by executing the command

.. code-block:: console

   hermes -p <parameter_dir> -o <output_dir>

The available argument options are

.. list-table:: CLI Argument Options
   :align: center

   * - -h, --help
     - Display the CLI help

   * - -p `<directory>`
     - Specify the parameter source directory

   * - -o `<directory>`
     - Specify the result output directory

   * - -s `<style>`
     - Specify the style of the result plots

   * - -t
     - Run the CLI in test mode

"""
import os
import shutil
import sys
import argparse
from typing import List, Optional

from ruamel.yaml.constructor import ConstructorError
from rich.console import Console

from hermespy.core.executable import Executable
from hermespy.core.factory import Serializable, Factory

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def hermes(args: Optional[List[str]] = None) -> None:
    """HermesPy Command Line Interface.

    Default entry point to execute hermespy `.yml` files via terminals.

    Args:

        args ([List[str], optional):
            Command line arguments.
            By default, the system argument vector will be interpreted.
    """

    # Recover command line arguments from system if none are provided
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='HermesPy - The Heterogeneous Mobile Radio Simulator',
                                     prog='hermes')
    parser.add_argument("-p", help="settings directory from which to read the configuration", type=str)
    parser.add_argument("-o", help="output directory to which results will be dumped", type=str)
    parser.add_argument("-s", help="style of result plots", type=str)
    parser.add_argument('-t', '--test', action='store_true', help='run in test-mode, does not dump results')
    parser.add_argument('-l', '--log', action='store_true', help='log the console information to a txt file')

    arguments = parser.parse_args(args)
    input_parameters_dir = arguments.p
    results_dir = arguments.o
    style = arguments.s

    # Create console
    console = Console(record=arguments.log)
    console.show_cursor(False)

    # Draw welcome header
    console.print("\n[bold green]Welcome to HermesPy - The Heterogeneous Radio Mobile Simulator\n")

    console.print(f"Version: {__version__}")
    console.print(f"Maintainer: {__maintainer__}")
    console.print(f"Contact: {__email__}")

    console.print("\nFor detailed instructions, refer to the documentation https://barkhausen-institut.github.io/hermespy")
    console.print("Please report any bugs to https://github.com/Barkhausen-Institut/hermespy/issues\n")

    # Validate command line parameters
    if not input_parameters_dir:
        input_parameters_dir = os.path.join(os.getcwd(), '_settings')

    elif not(os.path.isabs(input_parameters_dir)):
        input_parameters_dir = os.path.join(os.getcwd(), input_parameters_dir)

    console.log(f"Configuration will be read from '{input_parameters_dir}'")

    with console.status("Initializing Environment...", spinner='dots'):

        ##################
        # Import executable from YAML config dump
        factory = Factory()

        try:

            # Load serializable objects from configuration files
            serializables: List[Serializable] = factory.load(input_parameters_dir)

            # Filter out non-executables from the serialization list
            executables: List[Executable] = [s for s in serializables if isinstance(s, Executable)]

            # Abort execution if no executable was found
            if len(executables) < 1:

                console.log("No executable routine was detected, aborting execution", style="red")
                exit(-1)

            # For now, only single executables are supported
            executable = executables[0]

            # Configure executable
            if results_dir is None:
                executable.results_dir = Executable.default_results_dir()

            else:
                executable.results_dir = results_dir

        except ConstructorError as error:

            print("\nYAML import failed during parsing of line {} in file '{}':\n\t{}".format(error.problem_mark.line,
                                                                                              error.problem_mark.name,
                                                                                              error.problem,
                                                                                              file=sys.stderr))
            exit(-1)

        # Configure console
        executable.console = console

        # Configure style
        if style is not None:
            executable.style = style

        # Inform about the results directory
        console.log("Results will be saved in '{}'".format(executable.results_dir))

        # Dump current configuration to results directory
        if not arguments.test:
            shutil.copytree(input_parameters_dir, executable.results_dir, dirs_exist_ok=True)

    ##################
    # run simulation
    executable.execute()

    ###########
    # Goodbye :)
    console.log('Configuration executed. Goodbye.')

    # Save log
    if arguments.log:
        console.save_text(os.path.join(executable.results_dir, 'log.txt'))


if __name__ == "__main__":

    ################################################################
    # read command line parameters and initialize simulation folders
    hermes()
