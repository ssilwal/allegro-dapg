#!/bin/bash
#SBATCH --job-name=dapg-basic
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ss14499/runs/%j_%x.out
#SBATCH --error=/scratch/ss14499/runs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --mem=8G
#SBATCH -c 1

singularity exec --overlay  /scratch/ss14499/cnda/overlay-50G-10M.ext3:ro\
                 --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro\
                 /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif\
                 /bin/bash -c
"
source /ext3/env.sh
set -x
conda activate allegro-dapg
cd /home/ss14499/development/dapg/dapg/examples
[ ! -d "$2" ] mkdir $2
python job_script.py --output $2 --config $1 --project $3
" 
