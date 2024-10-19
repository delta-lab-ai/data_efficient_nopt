import os
import torch


tubelet_size = 2
ckpt_path = "./exp/training_checkpoints/"
ckpt_name = "ckpt.tar"

ckpt = torch.load(os.path.join(ckpt_path, ckpt_name), map_location=torch.device('cpu'))
ext = ckpt_name.split('.')[1]
ckpt_name = ckpt_name.split('.')[0]


for k, v in ckpt['model_state'].items():
    if "encoder.patch_embed.proj.weight" in k:
        # shape = v.shape # emb_dim, c_in, 1, kernel, kernel
        new_tensor = torch.cat([torch.clone(v) for _ in range(tubelet_size)], dim=2) / tubelet_size
        ckpt['model_state'][k] = new_tensor
    if "decoder.head.weight" in k:
        new_tensor = torch.cat([torch.clone(v) for _ in range(tubelet_size)], dim=0)
        ckpt['model_state'][k] = new_tensor
    if "decoder.head.bias" in k:
        new_tensor = torch.cat([torch.clone(v) for _ in range(tubelet_size)], dim=0)
        ckpt['model_state'][k] = new_tensor

torch.save(ckpt, os.path.join(ckpt_path, ckpt_name+"_expand."+ext))
