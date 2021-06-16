import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from config import config 


class ViralDataset(Dataset):
    def __init__(self, L, stride, max_len, min_len=0):
        super().__init__()
        data = np.load('data/viral.npz',allow_pickle=True)
        seqs = data['seq']
        lens = data['len']
        ids = data['id']
        self.seqs = seqs[(lens>=min_len)&(lens<max_len)]
        self.ids = ids[(lens>=min_len)&(lens<max_len)]
        self.lens = lens[(lens>=min_len)&(lens<max_len)]
        self.L = L
        self.alph = 4
        self.stride = stride
        self.max_len = max_len
        self.sid, self.pos = [], []
        for si,s in enumerate(self.seqs):
            for i in range(L,len(s),stride):
                self.pos.append(i-L)
                self.sid.append(si)
                
    def get_seq(self, i):
        si, idx = self.sid[i], self.pos[i]
        return self.seqs[si][idx:idx+self.L]
        

    def __len__(self):
        return len(self.sid)
    
    def __getitem__(self, i): 
        s = self.get_seq(i)
        X = torch.from_numpy(s).type(torch.int64)
        X = F.one_hot(X, num_classes=self.alph)
        X.transpose_(0,1)
        X = X.float()
        return X, (self.sid[i], self.pos[i])
    

# fully decompresss seqs before training 
# needs 20x more memory, but runs ~30% faster
class SeqDataset_decomp(Dataset):
    def __init__(self, seqs, ed, alph, L):
        super().__init__()
        self.ed = torch.tensor(ed, dtype=torch.float64) / L
        self.seqs = torch.tensor(seqs, dtype=torch.int64)
        self.X = F.one_hot(self.seqs, num_classes=alph)
        self.X.transpose_(2,3)
        self.X = self.X.float()

    def __len__(self):
        return len(self.ed)
    
    def __getitem__(self, i):   
        return self.X[i][0], self.X[i][1], self.ed[i]
    

class SeqDataset(Dataset):
    def __init__(self, seqs, ed, alph, L):
        super().__init__()
        self.ed = torch.tensor(ed, dtype=torch.float64) / L
        self.seqs = seqs
        self.alph = alph

    def __len__(self):
        return len(self.ed)
    
    def __getitem__(self, i):   
        X = torch.from_numpy(self.seqs[i]).type(torch.int64)
        X = F.one_hot(X, num_classes=self.alph)
        X.transpose_(1,2)
        X = X.float()
        return X[0], X[1], self.ed[i]

    
def load_mmseqs(src, N,L,alph):
    c = config['seqgen_dir']+"/{src}_{phase}_N{N}_L{L}_A{alph}.npz"
    train = np.load(c.format(phase="train",src=src, N=N,L=L,alph=alph))
    val = np.load(c.format(phase="val",src=src, N=N,L=L,alph=alph))
    return train, val 

    
def train_val_datasets(src, N, L, alph):
    train_data, val_data = load_mmseqs(src=src, N=N,L=L,alph=alph)

    train_dataset = SeqDataset(train_data['seqs'], train_data['ed'],alph=alph, L=L)
    val_dataset = SeqDataset(val_data['seqs'],val_data['ed'],alph=alph, L=L)
    
    return train_dataset, val_dataset