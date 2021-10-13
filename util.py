
import numpy as np

def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)



def evaluation_prf( scores_, targets_):
    n, n_class = scores_.shape
    Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        # targets[targets == -1] = 0
        Ng[k] = np.sum(targets == 1)
        Np[k] = np.sum(scores >= 0.5)
        Nc[k] = np.sum(targets * (scores >= 0.5))
    # Np[Np == 0] = 1
    if np.sum(Np) == 0:#避免分母为0
        Np[Np == 0] = 1
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)

    pred_y = np.where(scores_>=0.5,1,0)
    mismatches = pred_y ^ targets_
    hamming_loss = mismatches.sum()/(n*n_class)
    # CP = np.sum(Nc / Np) / n_class
    # CR = np.sum(Nc / Ng) / n_class
    # CF1 = (2 * CP * CR) / (CP + CR)
    return OP, OR, OF1,hamming_loss




