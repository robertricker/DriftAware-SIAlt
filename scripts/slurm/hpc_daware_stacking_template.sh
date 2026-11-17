#!/bin/bash
#
#SBATCH --job-name=cci_da_st
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1           
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=16             
#SBATCH --time=96:00:00               
#SBATCH --partition=comp
#SBATCH --mem-per-cpu=8G          

set -euo pipefail

module load Python/3.10.4-GCCcore-11.3.0

source /cluster/projects/105517-CCI+SeaIce/envs/driftaware/bin/activate

srun driftaware stacking "${1:?Pass a resolved configuration snapshot}"
