#!/bin/bash
#SBATCH --job-name=judge2       # Job name
#SBATCH --output=z_judge2_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_judge2_%j.err         # Error log
#SBATCH --time=06:00:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:8                     # Request 8 GPU
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
cd /projects/bfdz/zluo8/tool_and_judge2
source activate_environment.sh
cd /projects/bfdz/zluo8/tool_and_judge2
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface

# Run the training script

python judge.py --config judge_config_slurm2_vllm.py --num-gpus 8 --single-gpu-memory 141
python judge.py --config judge_config_slurm2_hf.py --num-gpus 8 --single-gpu-memory 141