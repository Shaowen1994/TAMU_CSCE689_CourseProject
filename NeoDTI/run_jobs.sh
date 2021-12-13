#!/bin/bash
# $(seq 0 41)
for ratio in {'one',} # 'ten', 'all'
do
for test in {'o','homo_protein_drug','drug','disease','sideeffect','unique'} #'o','homo_protein_drug','drug','disease','sideeffect','unique'
do
echo "#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:2
####SBATCH --partition=xlong

##OPTIONAL JOB SPECIFICATIONS
####SBATCH --mail-type=ALL
####SBATCH --mail-user=sunyuanfei@tamu.edu
#SBATCH --account=122807221847
#SBATCH --job-name=${ratio}_${test}
#SBATCH --output=/scratch/user/sunyuanfei/Projects/TAMU_CSCE689_CourseProject/NeoDTI/job_logs/${ratio}_${test}


#First Executable Line

module purge
source ~/.bashrc
module load cuDNN/5.0-CUDA-8.0.44
conda activate NeoDTI

python NeoDTI_cv.py -r $ratio -t $test

conda deactivate
module purge">run_jobs.job
sbatch run_jobs.job
sleep 1s
done
done