#!/bin/bash
#
#SBATCH --job-name=da_gridding     
#SBATCH --output=output.txt
#SBATCH --nodes=1           
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=16             
#SBATCH --time=96:00:00               
#SBATCH --partition=comp
#SBATCH --mem-per-cpu=8G          

module load Python/3.10.4-GCCcore-11.3.0

source /cluster/home/rori/venv/daware-venv/bin/activate

srun python /cluster/home/rori/python_projects/DriftAware-SIAlt/main.py /cluster/projects/108541-SO-SIMBA/python_projects/packages/DriftAware-SIAlt/config/config_hpc_gridding.yaml
