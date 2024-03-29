import numba as nb 
from numba import jit, njit
import numpy as np
import editdistance as ed
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

from config import config, grid_params


# check if rows of P sum to 1 
@njit
def check_validity(P):
    assert(np.sum((np.sum(P,axis=0)-1)**2)<1e-10 and "rows of P must sum up to 1")
    

# lambda is proportional to the mutation rate
@njit
def transition_probs(rate):
    profile = np.array([1,1,1,1])
    profile[0] = np.random.randint(0, int(1./rate))
    P = np.random.rand(4,4) 
    np.fill_diagonal(P, np.diag(P) * profile)
    P = P / np.sum(P,axis=0)
    check_validity(P)
    return P


# given transittion P, generate n mutations
def mutation_generator(P, n):
    X = np.random.randint(0, 4)
    mut = np.zeros(n, dtype=np.int8)
    for i in range(n):
        mut[i] = X
        X = np.random.choice(range(4),p=P[:,X])
    return mut


@njit
def seq_pair(s1, s2, alpha, mut):
    i, j = 0, 0
    for m in mut:
        a, b = np.random.randint(0, alpha,2)
        if m==0: # copy
            s1[i] = a
            s2[j] = a
            i, j = i+1, j+1
        elif m==1: # delte
            s1[i] = a
            i = i+1
        elif m==2: # insert
            s2[j] = b
            j = j+1
        elif m==3: # sub
            # make b different 
            b = (a + 1 + np.random.randint(0, alpha-1)) % alpha 
            assert(a != b)
            s1[i], s2[j] = a, b
            i, j = i+1, j+1
    return i, j


def gen_mmseqs(file, L, N, alpha, rate):
    edit_dists = np.zeros(N, dtype=np.int32)
    mut = np.zeros((N, L),dtype=np.int8)
    trans = np.zeros((N, 4, 4),dtype=np.float64)
    seqs = np.random.randint(alpha, size=(N, 2, L), dtype=np.int8)
    for i in tqdm(range(N),total=N):
        trans[i] = transition_probs(rate)
        mut[i] = mutation_generator(trans[i], L)
        seq_pair(seqs[i][0], seqs[i][1], alpha=alpha, mut=mut[i])
        edit_dists[i] = ed.eval(seqs[i][0],seqs[i][1])
    np.savez(file, seqs=seqs, trans=trans, mut=mut, ed=edit_dists)


            
def gen_shifted_seqs(file, L, N, alpha, rate):
    edit_dists = np.zeros(N, dtype=np.int32)
    lens = np.zeros((N,2), dtype=np.int32)
    shifts = np.zeros((N,2), dtype=np.int32)
    mut = np.zeros((N, L),dtype=np.int8)
    trans = np.zeros((N, 4, 4),dtype=np.float64)
    seqs = np.random.randint(alpha, size=(N, 2, L), dtype=np.int8)
    for i in tqdm(range(N),total=N):
        trans[i] = transition_probs(rate)
        mut[i] = mutation_generator(trans[i], L)
        lens[i] = seq_pair(seqs[i][0], seqs[i][1], alpha=alpha, mut=mut[i])
        shifts[i] = [np.random.randint(L+1-lens[i][j]) for j in range(2)]
        np.roll(seqs[i][0], shifts[i][0])
        np.roll(seqs[i][1], shifts[i][1])
        edit_dists[i] = ed.eval(seqs[i][0],seqs[i][1])
    np.savez(file, seqs=seqs, trans=trans, mut=mut, ed=edit_dists, shifts=shifts,lens=lens)
    
    
def gen_seqs(alg, phase, L, N, alpha, rate):
    algs = {
        'mmseqs': gen_mmseqs,
        'shseqs': gen_shifted_seqs,
    }
    file = f"{config['seqgen_dir']}/{alg}_{phase}_N{N}_L{L}_A{alpha}"
    algs[alg](file, L, N, alpha, rate)
    print(f"seqs saved in:\n{file}.npz")
        
        
if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument("--alg", type=str, nargs='+', help="generation algorihtm ", required=True)
    parser.add_argument("--phase", type=str, nargs='+', help="phases (train, valid, test)", default=["train", "val"])
    parser.add_argument("-N", nargs='+', type=int, help="number of seqs", default=[1000])
    parser.add_argument("-L", nargs='+', type=int, help="length of generated sequences", default=[16])
    parser.add_argument("-a", "--alpha", nargs='+', type=int, help="size of the alphabet", default=[4])
    parser.add_argument("-r", "--rate", nargs='+', type=float, help="rate for Markov Model", default=[0.01])
    grid = parser.parse_args()
    for params in grid_params(vars(grid)):
        gen_seqs(**params)
