# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from shutil import which
from subprocess import run
from sys import exit
from typing import Sequence
from tempfile import NamedTemporaryFile

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__SBATCH = """#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NUM_NODES
#SBATCH --exclusive
#SBATCH --ntasks=$NUM_NODES

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_port=6379

# Spinning up the head node
echo "Starting ray head at $head_node"
srun --disable-status --kill-on-bad-exit=0 --nodes=1 --ntasks=1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port --block &

# Spinning up worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting ray worker #$i at $node_i"
    srun --disable-status --kill-on-bad-exit=0 --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$head_node_ip:$head_node_port" --block &
done

# Running the original script
RAY_ADDRESS="$head_node_ip:$head_node_port" python $SCRIPT

# Cleanup after execution, i.e. stop the worker nodes
for ((i = SLURM_JOB_NUM_NODES - 1; i >= 0; i--)); do
    node_i=${nodes_array[$i]}
    echo "Stopping ray at $node_i"
    srun --disable-status --nodes=1 --ntasks=1 -w "$node_i" ray stop --force &
done
"""


def sHermes(args: Sequence[str] | None = None) -> None:
    """Command line tool for scheduling a HermesPy script within a SLURM cluster.

    Args:

        args: Sequence of command line arguments.
    """

    arg_parser = ArgumentParser(
        prog="HermesPy SLURM Submission Helper",
        description="A command line tool to help submit HermesPy simulation jobs to SLURM clusters with minimal effort.",
    )

    # Add argument for the actual script to be executed
    arg_parser.add_argument(
        "script", help="The python script containing the HermesPy simulation to be executed."
    )

    # Argument for the number of exclusively reserved nodes
    arg_parser.add_argument(
        "--num-nodes",
        "-n",
        type=int,
        default=1,
        help="Number of exclusive SLURM nodes to reserve for this HermesPy run.",
    )

    # Argument for skipping sanity checks like detecting the sbash command
    arg_parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip sanity checks before job submission to SLURM.",
    )
    # Parse command line arguments
    parsed_args = arg_parser.parse_args(args)

    # Sanity check: Scan for the required sbash binary
    # If it doesn't exist, assume we are not in a SLURM environment
    if not parsed_args.skip_checks and which("sbatch") is None:
        print("Synity check: sbash command not detected. Are you in a SLURM environment?")
        exit(-1)

    # Build sbatch
    batch_replacements = {
        "$JOB_NAME": parsed_args.script,
        "$NUM_NODES": str(parsed_args.num_nodes),
        "$SCRIPT": parsed_args.script,
    }

    batch_script = str(__SBATCH)
    for key, value in batch_replacements.items():
        batch_script = batch_script.replace(key, value)

    # Deploy the sBatch to slurm by writing a script to the temp directory
    # Call sbatch afterwards
    with NamedTemporaryFile() as temp_file:

        # Initialize the batch script
        temp_file.write(batch_script.encode("utf-8"))
        temp_file.flush()

        # Submit the job
        run(f"sbatch {temp_file.name}", shell=True, check=True)
