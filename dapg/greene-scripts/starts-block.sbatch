#!/bin/bash
#SBATCH --job-name=start
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ss14499/runs/%j_%x.out
#SBATCH --error=/scratch/ss14499/runs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH -c 1

singularity exec --overlay  /scratch/ss14499/cnda/her-overlay-50G-10M.ext3:ro\
                 --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro\
                 /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif\
                 /bin/bash -c "
source $/.bashrc
export PATH=/scratch/ss14499/miniconda3/envs/allegro-dapg/bin:/home/ss14499/.local/bin:/home/ss14499/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ss14499/.mujoco/mujoco200/bin
conda activate allegro-dapg
cd /home/ss14499/development/dapg/dapg/examples

echo \"dapg commit id:\" && git rev-parse HEAD

echo \"mj_allegro commit:\"
cat /ext3/miniconda3/envs/allegro-dapg/lib/python3.6/site-packages/mj_allegro_envs-1.0.0.dist-info/direct_url.json

echo \"mjrl commit:\"
cat /ext3/miniconda3/envs/allegro-dapg/lib/python3.6/site-packages/mjrl-1.0.0.dist-info/direct_url.json

echo \"allegro-dapg commit:\"
cat /ext3/miniconda3/envs/allegro-dapg/lib/python3.6/site-packages/allegro_ik_controller-1.0.0.dist-info/direct_url.json

cd ../../../mj_envs/ && echo \"mj_envs commit:\" && git rev-parse HEAD && cd -

[ ! -d $2 ] mkdir $2
python start_job_script.py --output $2 --config $1 --project $3
" 
