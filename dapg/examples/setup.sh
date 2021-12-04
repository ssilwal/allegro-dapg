singularity exec --overlay  /scratch/ss14499/cnda/overlay-50G-10M.ext3:ro\
                 --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro\
                 /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif\
                 /bin/bash 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ss14499/.mujoco/mujoco200/bin
python job_script.py --output $2 --config $1 --project $3
