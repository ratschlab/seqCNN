import numpy as np 
from tqdm import tqdm 
from numba import njit, jit


def project(emb, dout):
    din = emb.shape[1]
    G = np.random.randn(din, dout)
    p = np.matmul(emb, G)
    p = np.sign(p)
    return p

def projection_dist(P):
    prob_mismatch = 1-(np.matmul(P, P.transpose()) + Dout) / Dout / 2
    proj_dist = 1-np.cos(np.pi*prob_mismatch)
    return proj_dist


# Generalized N-dimensional products
def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def equal_pairs(a):
    ind = np.argsort(a)
    u, c = np.unique(a, return_counts=True)
    c = np.insert(c, 0, 0)
    cs = np.cumsum(c)

    cl = [ind[cs[j]:cs[j+1]] for j in range(len(cs)-1) if c[j+1]>1] 
    eq = [cartesian_product((x,x)) for x in cl]
    eq = np.concatenate(eq)
    return eq

def sensitive_hash_params(N, p1, p2, recall):
    m = int(np.log(N)/np.log(1/(1-p2)))
    P1 = (1-p1)**m
    M = int(np.log(1-recall)/np.log(1-P1))
    return m, M
    

def hash_binary(proj, m, M):
    N, D = proj.shape
    pr = (proj>0).astype(np.int64)
    pr = pr.transpose()
    
    v = 2**np.arange(m, dtype=np.int64)
    H = np.zeros((M,N),dtype=np.int64)
    for j in tqdm(range(M),total=M):
        rperm = np.random.permutation(D)[:m]
        H[j] = np.matmul(v, pr[rperm])
    return H

def nn_pairs(H):
    NN = np.concatenate([equal_pairs(h) for h in tqdm(H,total=len(H))])
    NN = NN[NN[:,1]<NN[:,0],:]
    return NN


def nn2index(NN, NN_dists):
    nn_index1 = np.stack((NN[:,0], NN_dists,NN[:,1]))
    nn_index2 = np.stack((NN[:,1], NN_dists,NN[:,0]))
    nn_index = np.concatenate((nn_index1,nn_index2),axis=1) # add reverse edges 
    ind = np.lexsort(nn_index)
    nn_index = nn_index[:,ind]
    unique, uind = np.unique(nn_index[2,:],return_index=True)
    uind = np.append(uind,len(unique))
    adj = dict()
    for ui,i in tqdm(enumerate(unique),total=len(unique)):
        x, y = uind[ui], uind[ui+1]
        adj[i.astype(np.int64)] = (nn_index[0,x:y].astype(np.int64),nn_index[1,x:y])
    return adj




@njit
def nn_dists(X, NN):
    NN_dists = [1-np.dot(X[i],X[j]) for i,j in NN]
    NN_dists = np.array(NN_dists)
    return NN_dists




    
def PR_curve(NN, T):
    np.fill_diagonal(T, 0)
    N = T.shape[0]
    P = np.zeros((N,N))
    P[NN[:,0],NN[:,1]] = 1
    P[NN[:,1],NN[:,0]] = 1
    recall = np.sum(P * T) / np.sum(T)
    precision = np.sum(P * T) / np.sum(P)
    
    return precision, recall 

@njit
def search(NN, NN_dists, samples):
    index = dict()
    for si, s in enumerate(samples):
        ind0, ind1 = NN[:,0]==s, NN[:,1]==s
        i, d = ( np.concatenate((NN[ind0,1],NN[ind1,0])),
                    np.concatenate((NN_dists[ind0],NN_dists[ind1])) )
        I = np.argsort(d)
        index[s] = (i[I], d[I]) 
    return index


def sample_PR(samples, T, index, k=None, th=None):
    I,J = np.nonzero(T)
    I = samples[I]
    neq = I!=J
    nn_pairs = np.array([I[neq],J[neq]]).transpose()
    
    recall = []
    dist_calls = 0
    for i, j in nn_pairs:
        if k:
            I = index[i][0][:k]
        elif th:
            I = index[i][0][index[i][1]<th]
        else:
            I = index[i][0]
        recall.append(j in I)
        dist_calls += len(I)

    return np.mean(recall), dist_calls


def nearest_neighbor(X, p1, p2, recall):
    N = X.shape[0]
    m, M = sensitive_hash_params(N, p1=p1, p2=p2, recall=recall)

    pr = project( X, 2*m*(int(np.log(M)+1) ) )
    H = hash_binary(pr, m=m, M=M)
    NN = nn_pairs(H)
    return NN
                       
                       
def nn_index(X, d1, d2, recall):
    print('building index ...')
    NN = nearest_neighbor(X, d1, d2, recall=recall)
    print('computing embed dists ... ')
    NN_dists = nn_dists(X, NN)
    return NN, NN_dists, 
