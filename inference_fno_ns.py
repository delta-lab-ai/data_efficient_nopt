'''
This code generates the prediction on one instance. 
Both the ground truth and the prediction are saved in a .pt file.
'''
import os
from unittest import result
import yaml
from collections import OrderedDict
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import linregress
from models.fno_ns import FNO3d

from data_utils.datasets_ns import KFDataset
from utils.losses import LpLoss
from utils.utils import count_params
from tqdm import tqdm
from pdb import set_trace as bp


@torch.no_grad()
def get_pred(args):
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.ckpt_path:
        save_dir = os.path.join('/'.join(args.ckpt_path.split('/')[:-1]), 'icl_results')
    else:
        basedir = os.path.join('exp', config['log']['logdir'])
        save_dir = os.path.join(basedir, 'icl_results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fno-prediction-demo%d.pt'%(args.num_demos if args.num_demos else 0))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prepare data
    dataset = KFDataset(paths=config['data']['paths'],
                        raw_res=config['data']['raw_res'],
                        data_res=config['data']['data_res'],
                        pde_res=config['data']['data_res'],
                        n_samples=config['data']['n_test_samples'],
                        total_samples=config['data']['total_test_samples'],
                        offset=config['data']['testoffset'],
                        t_duration=config['data']['t_duration'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    if args.num_demos:
        dataset_icl = KFDataset(paths=config['data']['paths'],
                            raw_res=config['data']['raw_res'],
                            data_res=config['data']['data_res'],
                            pde_res=config['data']['data_res'],
                            n_samples=config['data']['total_demo_samples'],
                            total_samples=config['data']['total_demo_samples'],
                            offset=config['data']['demo_offset'],
                            t_duration=config['data']['t_duration'])
        dataloader_icl = DataLoader(dataset_icl, batch_size=args.num_demos, shuffle=False, drop_last=False)
        assert args.num_demos <= config['data']['total_demo_samples']
        u_demos, a_in_demos = next(iter(dataloader_icl))
        u_demos = u_demos.to(device)
        a_in_demos = a_in_demos.to(device)

    # create model
    model = FNO3d(modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                modes3=config['model']['modes3'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers'], 
                act=config['model']['act'], 
                pad_ratio=config['model']['pad_ratio'],
                num_demos=args.num_demos).to(device)
    num_params = count_params(model)
    print(f'Number of parameters: {num_params}')
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        # model.load_state_dict(ckpt['model'])

        new_state_dict = OrderedDict()
        for key, val in ckpt['model'].items():
            name = key[7:] if 'module' in key else key
            new_state_dict[name] = val 
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in state and state[k].size() == new_state_dict[k].size()}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        message = model.load_state_dict(state)
        # self.model.load_state_dict(new_state_dict)
        unload_keys = [k for k in new_state_dict.keys() if k not in pretrained_dict]
        if len(unload_keys) > 0:
            import warnings
            warnings.warn("Warning: missing keys during restoring checkpoint: %s"%(str(unload_keys)))

        print('Weights loaded from %s' % args.ckpt_path)
    # metric
    lploss = LpLoss(size_average=True)
    model.eval()
    truth_list = []
    pred_list = []
    losses = []
    losses_normalized = []
    pbar = tqdm(dataloader, total=len(dataloader))
    # for u, a_in in dataloader:
    for u, a_in in pbar:
        u, a_in = u.to(device), a_in.to(device)
        if args.num_demos is None or args.num_demos == 0:
            out = model(a_in)
        else:
            model.target = u # for debugging purpose
            out = model.forward_icl(a_in, a_in_demos, u_demos, use_tqdm=args.tqdm)
        data_loss = lploss(out, u)
        data_loss_normalized = lploss(out / torch.abs(out).max(), u / torch.abs(u).max())
        losses.append(data_loss.item())
        losses_normalized.append(data_loss_normalized.item())
        # print(data_loss.item())
        truth_list.append(u.detach().cpu())
        pred_list.append(out.detach().cpu())

    # print(np.mean(losses))
    slope, intercept, r, p, se = linregress(torch.cat(pred_list, dim=0).view(-1).numpy(), torch.cat(truth_list, dim=0).view(-1).numpy())
    print("RMSE:", np.mean(losses), "RMSE (normalized)", np.mean(losses_normalized), "R2:", r, "Slope:", slope)
    truth_arr = torch.cat(truth_list, dim=0)
    pred_arr = torch.cat(pred_list, dim=0)
    if args.save_pred:
        torch.save({
            'truth': truth_arr,
            'pred': pred_arr,
            'rmse': np.mean(losses),
            'rmse_normalized': np.mean(losses_normalized),
            'r2': r,
            'slope': slope,
        }, save_path)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--num_demos', type=int, default=None)
    parser.add_argument('--tqdm', action='store_true', default=False, help='Turn on the tqdm')
    parser.add_argument('--save_pred', action='store_true', default=False, help='Save predictions')
    args = parser.parse_args()
    get_pred(args)
