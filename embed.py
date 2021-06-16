import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import editdistance

import torch 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from config import config 
import models


def load_pretrained_net( groups, kernel, num_layers, stride=2,in_channels=4, channels=1, i=0):
    paths = glob.glob(f"{config['networks_dir']}/*_in_channels{in_channels}_num_layers{num_layers}_channels{channels}_kernel{kernel}_stride{stride}_groups{groups}_*")
    print(paths)
    net = models.SeqCNN(in_channels=in_channels, groups=groups, kernel=kernel, num_layers=num_layers, stride=stride, channels=channels)
    net.load_state_dict(torch.load(paths[i]))
    return net
    
def embed_seqs(net, dataset, L, stride, device):
    loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)

    embeddings = []
    sids = []
    pos = []
    for data, (si,idx) in tqdm(loader,total=len(loader)):
        data = data.to(device)
        embed = net(data)
        embeddings.append(embed.cpu().data.numpy())
        sids.append(si)
        pos.append(idx)

    embeddings = np.concatenate(embeddings)
    embeddings = embeddings.squeeze()
    norms = np.linalg.norm(embeddings,axis=1)
    embeddings = np.divide(embeddings, norms[:,np.newaxis])
    
    sids = np.concatenate(sids)
    pos = np.concatenate(pos)
    return embeddings, sids, pos   


# make sure embeddings are unit norm 
def pairwise_embed_dist(embeddings):
    cos = np.matmul(embeddings, embeddings.transpose())
    return 1-cos


def pairwise_edit_dist(dataset):
    N = len(dataset)
    dists = np.zeros((N,N))
    for i in tqdm(range(N),total=N):
        for j in range(i+1,N):
            dists[i,j] = editdistance.eval(dataset.get_seq(i), dataset.get_seq(j))
    dists = dists + dists.transpose()
    return dists


def sample_edit_dist(dataset, num_samples):
    samples = np.random.permutation(len(dataset))[:num_samples]
    N = len(dataset)
    M = len(samples)
    dists = np.zeros((M,N))
    for si,i in tqdm(enumerate(samples),total=M):
        for j in range(N):
            dists[si,j] = editdistance.eval(dataset.get_seq(i), dataset.get_seq(j))
    return dists, samples
    
    
def get_dataframe(dataset, edit_dists, embed_dists):  
    inds = np.triu_indices(len(dataset),1)
    df = pd.DataFrame({'edit dist': edit_dists[inds], 
                       'embed dist': embed_dists[inds], 
                       's1': inds[0], 
                       's2': inds[1]})
    stats = df.groupby('edit dist').agg({'embed dist': 'count'}).reset_index()
    stats = stats.rename(columns={'embed dist': 'ed count'})
    stats['inv freq'] = 1/stats['ed count']
    df = pd.merge(df, stats, on="edit dist")
    return df

def get_stats(dataset, df, qs=[.1, .9]):
    funcs = [np.mean, np.std, np.median, 'count']
    for qi, q in enumerate(qs):
        quant = lambda x: np.quantile(x, q=q)
        quant.__name__ = f"q{qi}"
        funcs.append(quant)    

    stats = df.groupby('edit dist').agg({'embed dist': funcs}).reset_index()
    return stats
    
def get_samples(dataset, df, num_samples=100):
    samples = df.sample(dataset.L*num_samples, weights='inv freq')
    row2id = {i:dataset.ids[dataset.sid[i]] for i in range(len(dataset))}

    samples['name1'] = samples.apply(lambda x: f"{row2id[x.s1]}", axis=1)
    samples['name2'] = samples.apply(lambda x: f"{row2id[x.s2]}", axis=1)
    samples['text'] = samples.apply(lambda r: f"s1={r.name1}, s2={r.name2}",axis=1)
    return samples


def sparsity(edit_dists):
    ued, edcounts = np.unique(edit_dists[edit_dists>0].flatten(), return_counts=True)
    f, (ax2,ax1) = plt.subplots(1,2)
    f.set_figheight(5)
    f.set_figwidth(10)
    ax1.plot(ued, np.cumsum(edcounts)/edit_dists.shape[0])
    ax1.set_xlim(0, .5)
    ax1.set_ylim(0, 20)
    ax1.set_xlabel("edit dist threshold ")
    ax1.set_ylabel("average degree (edges / vertices)")
    ax1.grid('on', )
    ax2.plot(ued, np.cumsum(edcounts)/np.sum(edcounts))
    ax2.set_xlabel("edit dist threshold ")
    ax2.set_ylabel("sparsity (edges / edges of complete graph)")
    ax2.grid('on')
    f.savefig("results/sparsity.pdf", format="pdf")
    return f

def sample_edit_vs_embed_dists(edit_dists, embed_dists, neighbors):
    rinds = np.random.randint(0,edit_dists.shape[0],10)
    sub_sorted = np.take_along_axis(edit_dists[rinds], neighbors[rinds,1:50],axis=1)
    sub_sorted2 = np.take_along_axis(embed_dists[rinds], neighbors[rinds,1:50],axis=1)
    plt.figure(figsize=(7,7))
    f, (ax1,ax2) = plt.subplots(1,2)
    f.set_figheight(7)
    f.set_figwidth(14)
    ax1.plot(sub_sorted.transpose(), marker='.', alpha=.5)
    ax1.grid('on')
    ax1.set_xlabel('nearest neighbor (ordered by edit dist)')
    ax1.set_ylabel('edit distance')
    ax2.plot(sub_sorted.transpose(), sub_sorted2.transpose(),ls=':',marker='.', markersize=5,)
    ax2.set_ylabel('embeding distance')
    ax2.set_xlabel('edit distance')
    ax2.grid('on')
    ax1.legend([f"seq {i}" for i in rinds])


def embed_vs_ed(dataset, samples, x, y):
    def print_seq(trace, points, selector):
        pi = points.point_inds[0]
        name1, name2 = samples.name1.iloc[pi], samples.name2.iloc[pi]
        i, j = data.s1.iloc[pi], data.s2.iloc[pi]
        s1, s2 = dataset.get_seq(i), dataset.get_seq(j)
        s1, s2 = vec2seq(s1), vec2seq(s2)
        print(f"id 1 = {name1}, id2 = {name2}\n\n{s1}\n{s2}")
    trcol = lambda alpha: f'rgba(0,0,0,{alpha})'

    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=samples[x], 
                             y=samples[y],
                             mode='markers', 
                             marker_size=2,
                             name='sample'))

    fig.data[0].on_click(callback=print_seq)
    return fig
