#!/bin/sh

# SLURM options:

#SBATCH --job-name=ecc_dyn
#WBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=7-00:00:00

#SBATCH --mail-user=falxa@apc.in2p3.fr
#SBATCH --mail-type=END,FAIL

# Commands to be submitted:

date

# singularity exec --bind /work/,/home/falxa/ /work/falxa/singularity/EPTA_ENTERPRISE.sif python3 /home/falxa/scripts/nonstat/get_evidences_filter.py
singularity exec --bind /work/,/home/falxa/ /work/quelquejay/PTA/MY_ENTERPRISE.sif python3 /home/falxa/scripts/nonstat/real_data_test/get_evidences_ecc.py --ecc $1

date