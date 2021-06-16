from numba import njit
import numpy as np


@njit
def editDist(str1, str2):
    m, n = len(str1), len(str2)
    dp = np.zeros((m+1,n+1),np.int32)
 
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    return dp[m][n]

@njit
def numba_choose(ind, p):
    if (p==0) or (p==-1):
        return ind[p]
    else:
        return ind[np.random.randint(len(ind))] 

@njit
def align(str1: str, str2: str, p: int = 2):
    m, n = len(str1), len(str2)
    dp = np.zeros((m+1,n+1),np.int64)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    # actions 0: copy, 1: sub, 2: ins, 3: del
    actions = [] 
    i, j, di, dj = m, n, (-1, -1, 0, -1), (-1, -1, -1, 0)
    while i>0 and j>0:
        mismatch = (str1[i-1]!=str2[j-1])*1
        choices = np.array((dp[i-1,j-1]+4*mismatch,   # copy 
                            dp[i-1,j-1]+1, # subsitute 
                            dp[i,j-1]+1,   # insert
                            dp[i-1,j]+1))  # delete 
        choices = np.nonzero(dp[i,j]==choices)[0]
        action = numba_choose(choices, p=0)
        actions.append(action)
        i, j = i+di[action],j+dj[action]
    actions.extend([2]*j)
    actions.extend([3]*i)
    actions = np.array(actions)[-1::-1]
    x, y, i, j = '', '', 0, 0
    for a in actions:
        if a!=2:
            x += str1[i]
            i += 1
        else:
            x += '-'
        if a!=3:
            y += str2[j]
            j += 1
        else:
            y += '-'
    return dp[m][n], actions, x, y


def edit_ops_desc(edits):
    ed2name = {0: 'c', 1: 's', 2: 'i', 3: 'd'}
    op,cnt = np.unique(edits, return_counts=True)
    op = {ed2name[o]:cnt[i] for i,o in enumerate(op)}
    return op


def vec2seq(s, Alph="acgt"):
    return "".join([Alph[x] for x in s])


def align_viewer(seq1, seq2, col_width = 100, edit_markers=" .><", p=2):
    s1, s2 = vec2seq(seq1), vec2seq(seq2)
    ed, edits, x, y = align(s1, s2,p) 
    print(edit_ops_desc(edits))

    for i in range(0,len(s1), col_width):
        l1, l2 = i, i+col_width
        print("")
        print(vec2seq(edits,edit_markers)[l1:l2])
        print(x[l1:l2])
        print(y[l1:l2])
        
        
        