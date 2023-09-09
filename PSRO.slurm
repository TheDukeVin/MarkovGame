#!/bin/bash
#SBATCH -c 1 # Number of cores requested
#SBATCH -t 3000 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=30000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o psro_std.out # Standard out goes to this file
#SBATCH -e psro_std.err # Standard err goes to this filehostname
python3 PSRO.py