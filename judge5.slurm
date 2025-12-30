#!/bin/bash
#SBATCH --job-name=judge5       # Job name
#SBATCH --output=z_judge5_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_judge5_%j.err         # Error log
#SBATCH --time=2:00:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
# #SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:4                     # Request 4 GPUs
#SBATCH --cpus-per-task=8                # CPU cores (increased from 4)
#SBATCH --mem=64G                        # RAM (increased from 32G)
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
cd /projects/bfdz/zluo8/tool_and_judge2
source activate_environment.sh
cd /projects/bfdz/zluo8/tool_and_judge2
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface

# Run the training script

python judge.py --config judge_config_slurm5_vllm.py --num-gpus 4 --single-gpu-memory 40
python judge.py --config judge_config_slurm5_hf.py --num-gpus 4 --single-gpu-memory 40