import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from plotly import graph_objects as go

from code.dataset import ViralDataset
from code import embed 
from code import sequence
    

def get_embedings(num_layers = 8, kernel=3, groups=4, device='cuda', max_len=1500, min_len=0, stride_ratio=1./4):
    L = 2**num_layers
    stride = int(L*stride_ratio)

    dataset = ViralDataset(min_len=min_len, max_len=max_len, L=L, stride=stride)
    net = embed.load_pretrained_net(groups=groups, kernel=kernel, num_layers=num_layers,)
    net = net.to(device)
    embeddings, sids, pos = embed.embed_seqs(net, dataset, L, stride, device)
    
    return dataset, embeddings

def get_edit_dists(dataset):
    # edit_dists = pairwise_edit_dist(dataset)
    # np.savez(f'viral_ed_L{L}_max_len{max_len}', edit_dists=edit_dists)
    data = np.load(f"data/viral_ed_L{dataset.L}_max_len{dataset.max_len}.npz", allow_pickle=True)
    edit_dists = data['edit_dists']
    print(f"calculated for {len(dataset)} substrings, forming {np.sum(edit_dists>0)} pairs")
    return edit_dists
    
    
def target_summary(embeddings, samples, dataset, target, index, print_info = True, max_sh=1):   
    I0, J = np.nonzero(target)
    I = samples[I0]
    neq = J!=I
    I0, I, J = I0[neq], I[neq], J[neq]
    dataset.sid = np.array(dataset.sid)
    dataset.pos = np.array(dataset.pos)
    info = [list(zip(dataset.sid[ind],dataset.pos[ind], ind)) for ind in [I,J]]
    info = np.array(info)


    df = pd.DataFrame(columns=["embed_dist",
                               "ed", "ed_min", "ed_sh", 
                               "hd", "hd_min", "hd_sh", 
                               "indexed",
                               "sample", 
                               "i1", "i2", 
                               "id1", "id2", 
                               "pos1", "pos2"])
    for i in trange(info.shape[1]):
        ed_min,hd_min = None, None
        for sh in range(-max_sh, max_sh):
            id1,pos1,i1 = info[0][i]
            id2, pos2,i2 = info[1][i]
            seq1 = dataset.seqs[id1][pos1:pos1+dataset.L]
            seq2 = dataset.seqs[id2][pos2+sh:pos2+sh+dataset.L]
            if len(seq1)!=len(seq2):
                continue
            ed = sequence.editDist(seq1,seq2)
            hd = np.sum(seq1!=seq2)
            if sh==0:
                ed_org = ed
                hd_org = hd
                s1, s2 = sequence.vec2seq(seq1),sequence.vec2seq(seq2)
                _, edits, _, _ = sequence.align(s1, s2,p=2) 
                op = sequence.edit_ops_desc(edits)
                desc = "\t".join([f"{k}:{v}" for k,v in op.items()])
            if hd_min==None or hd<hd_min:
                hd_min, hd_sh = hd, sh
            if ed_min==None or ed<ed_min:
                ed_min, ed_sh = ed, sh
        dist = 1-np.dot(embeddings[i1],embeddings[i2])
        df.loc[len(df),:] = (dist, 
                             ed_org, ed_min, ed_sh, 
                             hd_org, hd_min, hd_sh, 
                             (i2 in index[i1][0])*1,
                             i, i1, i2, id1, id2, pos1, pos2)
        if print_info:
            print(f"\n{i}\n"
                  f"{i1} | {dataset.ids[id1][:40]} | {id1} | {pos1} \n"
                  f"{i2} | {dataset.ids[id2][:40]} | {id2} | {pos2} \n"
                  f"{desc}\n"
                  f"{ed_org}\t{ed_min}\t{ed_sh}\n"
                  f"{hd_org}\t{hd_min}\t{hd_sh}")
    return df


def embed_vs_ed(dataset, samples, x, y):
    def print_seq(trace, points, selector):
        pi = points.point_inds[0]
        name1, name2 = samples.name1.iloc[pi], samples.name2.iloc[pi]
        i, j = data.s1.iloc[pi], data.s2.iloc[pi]
        s1, s2 = dataset.get_seq(i), dataset.get_seq(j)
        s1, s2 = sequence.vec2seq(s1), vec2seq(s2)
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