import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import importlib 
from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from config import config, grid_params
import training
import seq_models
from seq_models import SeqCNN


def fig2image(fig=None):
    if fig==None:
        plt.savefig('/tmp/seqCNN.png', dpi=200)
    else:
        fig.savefig('/tmp/seqCNN.png', dpi=200)
    img = cv2.imread('/tmp/seqCNN.png',cv2.COLOR_BGR2RGB)
    img = img.transpose(2,0,1)
    img = img[[2,1,0],:,:]
    return img

def visualize_sample(model, loader, device, samples=20000):
    test_loader = DataLoader(dataset = loader.dataset, batch_size = samples,shuffle=False)
    data1, data2,target = next(iter(test_loader))
    embed1,embed2, target = model(data1.to(device)), model(data2.to(device)), target.to(device)
    dist = cosine_dist(embed1,embed2)
    loss = log_ratio_loss(dist, target)
    y=dist.cpu().data.numpy()
    x=target.cpu().data.numpy()
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x=x, y=y, s=1,alpha=.3)
    ax = plt.gca()
    ax.set_xlim(0,x.max())
    ax.set_ylim(0,y.max())
    ax.set_xlabel("edit dist")
    ax.set_ylabel("embed dist")
    ax.set_title(f"sample val loss {loss:.4f}")
    return fig2image(fig)

def cosine_dist(embed1, embed2, eps=1e-2):
    embed_sim = torch.nn.CosineSimilarity()
    l1loss = nn.L1Loss()
    return 1-embed_sim(embed1.flatten(1),embed2.flatten(1))
    
def log_ratio_loss(dist, target, eps=1e-2):
    l1loss = nn.L1Loss()
    return l1loss(torch.log(dist+eps),torch.log(target+eps))

def params_summary(name, params):
    def data_summary(src, N, L, alph):
        return f"{src}_alph{alph}_L{L}_N{N}"
    def model_summary(**kwargs):
        return "_".join(f"{k}{v}" for k,v in kwargs.items())
    data_desc = data_summary(**params['dataset_params'])
    model_desc = model_summary(**params['model_params'])
    timestamp = int(time.time())
    trWriter = SummaryWriter(f"{config['runs_dir']}/{name}/train_{data_desc}_{model_desc}_{timestamp}")
    valWriter = SummaryWriter(f"{config['runs_dir']}/{name}/val_{data_desc}_{model_desc}_{timestamp}")
    
    net_path = f"{config['networks_dir']}/seqCNN_{model_desc}_{timestamp}" 
    return trWriter, valWriter, net_path

def default_params():
    return dict(
        batchsize = 512,
        lr = 0.001,
        epochs = 100,
        device = 'cuda',
        patience = 1,
        min_delta = 1e-3,
        embed_dist = cosine_dist,
        criteria = log_ratio_loss,
        net_model = SeqCNN,
        visualize = visualize_sample,
        dataset_params = dict(
            src = 'shseqs',
            N = 10**6, 
            L = 512, 
            alph = 4),
        model_params = dict(
            in_channels = 4,
            num_layers = 9,
            channels = 1, 
            kernel = 5,
            stride = 2,
            groups = 1))


def run(name, src, stride, groups, layers, kernel):
    params = default_params()
    params['dataset_params'].update({'L': 2**layers, 
                                     'src': src})
    params['model_params'].update({'num_layers': layers,
                                   'kernel': kernel, 
                                   'stride': stride, 
                                   'groups': groups, 
                                   'num_layers': layers})
    trWriter, valWriter, net_path = params_summary(name, params)
    net, train_loader, val_loader = training.train_model(**params, trWriter=trWriter, valWriter=valWriter)
    print(f"saved network in {net_path}")
    torch.save(net.state_dict(), net_path )
    return net, train_loader, val_loader

        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument("--name", help="name of the experiment", default="test")
    parser.add_argument("--src", type=str, help="source of sequence data", default="shseqs")
    parser.add_argument("--layers", type=int, nargs='+', help="number of layers for CNN", default=[5])
    parser.add_argument("--groups", type=int, nargs='+', help="group size for CNN", default=[1])
    parser.add_argument("--stride", type=int, nargs='+', help="stride for CNN", default=[2])
    parser.add_argument("--kernel", type=int, nargs='+', help="kernel size for CNN", default=[5])
    grid = parser.parse_args()
    for pars in grid_params(vars(grid)):
        print(pars)
        run(**pars)
