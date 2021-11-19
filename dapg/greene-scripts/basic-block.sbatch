singularity exec --overlay  /scratch/ss14499/cnda/overlay-50G-10M.ext3:rw                     --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro                     /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c
"
source /ext3/env.sh
conda activate allegro-dapg
cd /home/ss14499/development/dapg/dapg/examples
python job_script.py --output $SCRATCH/experiments/test-job --config ./dapg-start.txt --project greene
 
" 
