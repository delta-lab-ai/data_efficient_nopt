# Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning ([Paper Link](https://arxiv.org/abs/2402.15734))

<img src="[teaser.png](https://github.com/user-attachments/assets/e523f36c-b261-4ae3-a5d4-4a141c746fe4)" width="200"/>

This repository contains code for pre-training and fine-tuning the Fourier Neural Operator and Video-MAE models with unlabeled, domain-specific data.

## Environment
Please use the following command to install the necessary packages to run the code.
```
pip install -r requirements.txt
```

## Data
### Data Generation
We have an additional repository [data_generation](https://github.com/jsong2333333/data_generation) that compiles all the data generation pipelines we utilize. A more comprehensive guideline can be found under the repository.
- For Poisson and Helmholtz data, we generate following **Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior** ([Github Link](https://github.com/ShashankSubramanian/neuraloperators-TL-scaling)).
- For Diffusion-Reaction and Incompressible Navier Stokes, we use the data from **PDEBench** ([Data Download Link](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)).
### Dataloader
Dataloaders can be found at `data_utils/`. This repository currently contains dataloader for the following PDE equations: Poisson, Helmholtz, Diffusion-Reaction and Incompressible Navier Stokes.
- For Poisson and Helmholtz data, we refer to the dataloader from **Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior** ([Github Link](https://github.com/ShashankSubramanian/neuraloperators-TL-scaling)).
- For Diffusion-Reaction and Incompressible Navier Stokes, we refer to the dataloader from **Multiple Physics Pretraining for Physical Surrogate Models** ([Github Link](https://github.com/PolymathicAI/multiple_physics_pretraining)).

<!---
## Training and Inference
- Configuration files (in YAML format) are in `configs/` for different PDE systems. For example, config for Poisson's is in `configs/operator_poisson.yaml`.  The main configs for the three systems are ``poisson-scale-k1_5``, ``ad-scale_adr0p2_1`` and ``helm-scale-o1_10``. The data paths and scales paths need to be set here. For example, the config at [configs/operator_poisson.yaml](config/operator_poisson.yaml) has the data setup and minimal hyperparameters as follows:
 ```
poisson-scale-k1_5: &poisson_scale_k1_5  # sampled eigenvalues are in (1,5) for diffusion
	  <<: *poisson
	  ... # can change other default configs from poisson if needed #
	  ...
	  train_path:  # path to train data
	  val_path:    # path to validation data
	  test_path:   # path to test data
	  scales_path: # path to train data scales for input normalization 
	  batch_size:       # batch size for training 
	  valid_batch_size: # batch size for validation
	  log_to_wandb:     # switch on for logging to weights&biases
	  mode_cut:         # number of fourier modes to use
	  embed_cut:        # embedding dimension of FNO
	  fc_cut:           # multiplier for last fc layer
``` 
- Data, trainer, and other miscellaneous utilities are in `utils/`. We use standard PyTorch dataloaders and models wrapped with DDP for distributed data-parallel training with checkpointing.
- The FNO model is the standard model and is in `models/`. The hyperparameters used are in the config files for the respective PDE systems.
- Environment variables for DDP (local rank, master port etc) are set in `export_DDP_vars.sh` to be sourced before running any distributed training. See the [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details on using DDP. There are other ways to implement this, but our run script is specifically for slurm systems.
- Example running scripts are in `run.sh` (4 GPU DDP train script). ``train.py`` is the training script (``utils/trainer.py``) is the trainer class and ``eval.py`` can be used for inference (``utils/inferencer.py`` is the inference class). See the run scripts for details.

  ```bash
bash run_gen_data.sh
python utils/get_scale.py

CUDA_VISIBLE_DEVICES=0 python train.py --yaml_config=./config/operators_poisson.yaml --config=poisson-scale-k1_5-k2.5_7.5-demo_3 --run_num=bs16_lr1.25e-4_sub100_attn_aug_12 --root_dir=./

CUDA_VISIBLE_DEVICES=0 python eval.py --yaml_config=./config/operators_poisson.yaml --config=poisson-scale-k1_5_demo_7_bs128 --run_num=test --root_dir=/ssd1/chenwy/Mahoney_SciML/neuraloperators-TL-scaling/poisson_demo_7_bs128_2 --weights=poisson_demo_7_bs128/expts/poisson-scale-k1_5_demo_7_bs128/test2/checkpoints/ckpt_best.tar

# can be used to generate feature data
# /pscratch/sd/c/chenwy/neuraloperators-TL-scaling/expts/poisson-scale-k1_5-k10_20/64x64_bs128_lr1e-3_baseline_0927/checkpoints/ckpt_best.tar
CUDA_VISIBLE_DEVICES=0 python eval.py --yaml_config=./config/operators_poisson.yaml --config=poisson-scale-k1_5-k2.5_7.5 --run_num=test --root_dir=/pscratch/sd/c/chenwy/neuraloperators-TL-scaling --weights=/pscratch/sd/c/chenwy/neuraloperators-TL-scaling/expts/poisson-scale-k1_5-k2.5_7.5/64x64_bs128_lr1e-3_baseline_0915/checkpoints/ckpt_best.tar
```
-->
