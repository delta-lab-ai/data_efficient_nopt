#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J vmae
#SBATCH --mail-user=
#SBATCH --mail-type=all
#SBATCH -t 6:00:00
#SBATCH -A 

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6


# run_name="fno_finetune_debug_m0" 
# config="pois-64-e5_15_ft0"   
# yaml_config="./config/operators_poisson.yaml"


run_name="fno_helm_ft5_r2_finetune" 
config="helm-64-o5_15_ft5_r2"   
yaml_config="./config/operators_helmholtz.yaml"

export MASTER_ADDR=$(hostname)

module load conda
conda activate conda_env

# number of gpus
ngpu=4

# run command
cmd="python train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config" # --use_ddp"
$cmd

# source DDP vars first for data-parallel training (if not srun, just source and then run cmd; see pytorch docs for DDP)
# srun -l -n $ngpu --cpus-per-task=10 --gpus-per-node $ngpu bash -c "source export_DDP_vars.sh && $cmd"