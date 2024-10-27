from datetime import datetime
import os
import yaml
import random
from argparse import ArgumentParser
import math
from tqdm import tqdm

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.fno_ns import FNO3d

from utils.losses import LpLoss
from data_utils.datasets_ns import KFDataset, KFaDataset, sample_data
from utils.utils import save_ckpt, count_params, dict2str
from pdb import set_trace as bp

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def eval_ns(model, val_loader, criterion, device):
    model.eval()
    val_err = []
    for u, a in val_loader:
        u, a = u.to(device), a.to(device)
        out = model(a)
        val_loss = criterion(out, u)
        val_err.append(val_loss.item())

    N = len(val_loader)

    avg_err = np.mean(val_err)
    std_err = np.std(val_err, ddof=1) / np.sqrt(N)
    return avg_err, std_err


def train_ns(model, 
             train_u_loader,        # training data
             train_a_loader,        # initial conditions
             val_loader,            # validation data
             optimizer, 
             scheduler,
             device, config, args):
    start_iter = config['train']['start_iter']
    v = 1/ config['data']['Re']
    t_duration = config['data']['t_duration']
    save_step = config['train']['save_step']
    eval_step = config['train']['eval_step']

    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']
    # set up directory
    base_dir = os.path.join('exp', "train", config['log']['logdir'], args.name)
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)
    
    S = config['data']['pde_res'][0]
    # set up wandb
    if wandb and args.log:
        run = wandb.init(project=config['log']['project'], 
                         name="train_" + config['log']['logdir'] + '_' + args.name,
                         entity=config['log']['entity'], 
                         group=config['log']['group'], 
                         config=config, reinit=True, 
                         settings=wandb.Settings(start_method='fork'))
    
    pbar = range(start_iter, config['train']['num_iter'])
    if args.tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    u_loader = sample_data(train_u_loader)
    a_loader = sample_data(train_a_loader)

    for e in pbar:
        log_dict = {}

        optimizer.zero_grad()
        # data loss
        if xy_weight > 0:
            _data = next(u_loader)
            u, a_in = _data
            u = u.to(device)
            a_in = a_in.to(device)
            out = model(a_in)
            data_loss = lploss(out, u)
        else:
            data_loss = torch.zeros(1, device=device)

        loss_ic = loss_f = 0.0
        loss = data_loss * xy_weight + loss_f * f_weight + loss_ic * ic_weight

        loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict['train loss'] = loss.item()
        log_dict['data'] = data_loss.item()
        if e % eval_step == 0:
            eval_err, std_err = eval_ns(model, val_loader, lploss, device)
            log_dict['val error'] = eval_err
        
        if args.tqdm:
            logstr = dict2str(log_dict)
            pbar.set_description(
                (
                    logstr
                )
            )
        if wandb and args.log:
            wandb.log(log_dict)
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer, scheduler)

    # clean up wandb
    if wandb and args.log:
        run.finish()


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # create model 
    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio']).to(device)
    # model = FNO2dWithBackbone(4, modes1=12, modes2=12, width=20, initial_step=1)
    num_params = count_params(model)
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')
    # Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        ckpt_model_dict = ckpt['model'] if 'model' in ckpt else ckpt['model_state_dict']
        for key, val in ckpt_model_dict.items():
            name = key
            if "decoder" in name: continue
            if "module" in key:
                name = name[7:]
            if "encoder" in name:
                name = name.replace("encoder", "backbone")
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
            warnings.warn("Warning: unload keys during restoring checkpoint: %s"%(str(unload_keys)))
        # model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    if args.test:
        batchsize = config['test']['batchsize']
        testset = KFDataset(paths=config['data']['paths'], 
                            raw_res=config['data']['raw_res'],
                            data_res=config['test']['data_res'], 
                            pde_res=config['test']['data_res'], 
                            n_samples=config['data']['n_test_samples'], 
                            offset=config['data']['testoffset'], 
                            t_duration=config['data']['t_duration'])
        testloader = DataLoader(testset, batch_size=batchsize, num_workers=4)
        criterion = LpLoss()
        test_err, std_err = eval_ns(model, testloader, criterion, device)
        print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')
    else:
        # training set
        batchsize = config['train']['batchsize']
        u_set = KFDataset(paths=config['data']['paths'], 
                          raw_res=config['data']['raw_res'],
                          data_res=config['data']['data_res'], 
                          pde_res=config['data']['data_res'], 
                          n_samples=config['data']['n_data_samples'], 
                          offset=config['data']['offset'], 
                          t_duration=config['data']['t_duration'])
        u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=8, shuffle=True)

        a_set = KFaDataset(paths=config['data']['paths'], 
                           raw_res=config['data']['raw_res'], 
                           pde_res=config['data']['pde_res'], 
                           n_samples=config['data']['n_a_samples'],
                           offset=config['data']['a_offset'], 
                           t_duration=config['data']['t_duration'])
        a_loader = DataLoader(a_set, batch_size=batchsize, num_workers=8, shuffle=True)
        # val set
        valset = KFDataset(paths=config['data']['paths'], 
                           raw_res=config['data']['raw_res'],
                           data_res=config['test']['data_res'], 
                           pde_res=config['test']['data_res'], 
                           n_samples=config['data']['n_test_samples'], 
                           offset=config['data']['testoffset'], 
                           t_duration=config['data']['t_duration'])
        val_loader = DataLoader(valset, batch_size=batchsize, num_workers=8)
        print(f'Train set: {len(u_set)}; Test set: {len(valset)}; IC set: {len(a_set)}')
        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])
        if args.resume:
            ckpt = torch.load(ckpt_path)
            optimizer.load_state_dict(ckpt['optim'])
            scheduler.load_state_dict(ckpt['scheduler'])
            config['train']['start_iter'] = scheduler.last_epoch
        train_ns(model, 
                 u_loader, a_loader, 
                 val_loader, 
                 optimizer, scheduler, 
                 device, 
                 config, args)
    print('Done!')
        
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--name', type=str, default="", help='name of this run')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='also load optimizer and scheduler')
    parser.add_argument('--test', action='store_true', help='Test')
    parser.add_argument('--tqdm', action='store_true', help='Turn on the tqdm')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    import datetime
    args.name = args.name + "--" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    subprocess(args)
