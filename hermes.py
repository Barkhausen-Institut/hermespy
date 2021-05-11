#!/usr/bin/env python

"""This function calls the simulator main function.

It loads and checks all input parameters, creates results directories, initializes random numbers seeds, adds the
simulation paths, generates, saves and plots the statistics.

It can be called as follows:

Example:
    $ python hermes.py

    uses default parameter (../_settings) and output directories (_results_yyyy-mm-dd_iii)

    $ python hermes.py -o <output_dir> -p <parameter_dir>

    uses user-defined parameter and output directories

    $python hermes.py -h

    help on usage


@authors: Andre Noll Barreto (andre.nollbarreto@barkhauseninstitut.org)
Copyright (C) 2019 Barkhausen Institut gGmbH
Released under the Gnu Public License Version 3

"""
import os
import shutil
import datetime
import sys
import argparse
from typing import List

from parameters_parser.parameters import Parameters
from simulator_core.random_streams import RandomStreams
from scenario.scenario import Scenario
from simulator_core.drop_loop import DropLoop


def hermes(args: List[str]) -> None:
    print("Welcome to HermesPy")
    parser = argparse.ArgumentParser(
        description="usage: hermes.py -p <settings_dir> -o <output_dir>")
    parser.add_argument("-p", help="Settings directory.")
    parser.add_argument("-o", help="Output directory.")
    arguments = parser.parse_args(args)
    input_parameters_dir = arguments.p
    results_dir = arguments.o

    # validate commandline parameters
    if not input_parameters_dir:
        input_parameters_dir = os.path.join(os.getcwd(), '_settings')

    print('Parameters will be read from ' + input_parameters_dir)

    if not results_dir:
        today = str(datetime.date.today())

        dir_index = 0
        results_dir = os.path.join(
            os.getcwd(),
            "results",
            today +
            '_' +
            '{:03d}'.format(dir_index))

        while os.path.exists(results_dir):
            dir_index += 1
            results_dir = os.path.join(
                os.getcwd(),
                "results",
                today +
                '_' +
                '{:03d}'.format(dir_index))

    print('Results will be saved in ' + results_dir)

    shutil.copytree(input_parameters_dir, results_dir)

    #########################
    # initialize parameters
    parameters: Parameters = Parameters(results_dir)
    parameters.read_params()

    ######################################
    # initialize random number generation
    random_number_gen = RandomStreams(parameters.general.seed)

    ##################
    # run simulation
    scenario = Scenario(
        parameters.scenario,
        parameters.general,
        random_number_gen)

    simulation_loop = DropLoop(parameters.general, scenario)
    statistics = simulation_loop.run_loop()

    statistics.save(results_dir)

    print('results saved in ' + results_dir)


if __name__ == "__main__":

    ################################################################
    # read command line parameters and initialize simulation folders
    hermes(sys.argv[1:])
