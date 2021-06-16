import numpy as np
from tqdm import tqdm 
from seq_dataset import ViralDataset
import editdistance
from config import config 
import argparse




def triangle(f1, f2, L, max_seq_len, stride, mod):
    data = np.load('data/viral.npz',allow_pickle=True)
    seqs = data['seq']
    lens = data['len']
    seqs = seqs[lens<max_seq_len*10000]
    ids = data['id']
    dataset = ViralDataset(seqs=seqs, L=L, stride=stride)
    N = len(dataset)
    dists = []
    for i in tqdm(range(f1,N, mod),total=int(N/mod)):
        for j in range(f2,N, mod):
            d = editdistance.eval(dataset.get_seq(i), dataset.get_seq(j))
#             dists.append( (i,j, d) )
    dists = np.stack(dists)
    np.savez(f"{config['viral_benchmark']}/viral{max_seq_len}_L{L}_s{stride}_{mod}_{f1}_{f2}", dists=dists)
    

def line(f1, L, max_seq_len, stride, mod):
    data = np.load('data/viral.npz',allow_pickle=True)
    seqs = data['seq']
    lens = data['len']
    seqs = seqs[lens<max_seq_len*10000]
    ids = data['id']
    dataset = ViralDataset(seqs=seqs, L=L, stride=stride)
    N = len(dataset)
    dists = np.zeros(N)
    for i in tqdm(range(N),total=int(N)):
        dists[i] = editdistance.eval(dataset.get_seq(i), dataset.get_seq(j))
    dists = np.stack(dists)
    np.savez(f"{config['viral_benchmark']}/viral{max_seq_len}_L{L}_s{stride}_{mod}_{f1}", dists=dists)

    
if __name__=='__main__':  
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument("-L", type=int, help="length of substrings", default=256)
    parser.add_argument("-s", type=int, help="stride of seqs", default=64)
    parser.add_argument("--max-len", type=int, help="maximum length of seq in 1000 basepair ", default=10)
    parser.add_argument("-m", type=int, help="mod for computation", required=True)
    parser.add_argument("-i", type=int, help="modulus position for first seq", required=True)
    parser.add_argument("-j", type=int, help="modulus position for first seq", required=True)
    params = parser.parse_args()
    
    triangle(f1=params.i, 
             f2=params.j, 
             L=params.L, 
             max_seq_len=params.max_len, 
             stride=params.s, 
             mod=params.m)