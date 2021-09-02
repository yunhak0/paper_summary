import os
import re
import numpy as np
from scipy import sparse
from SVD_integrated import SVD, SVDpp, SVD_integrated

if __name__ == '__main__':
    movielens_dir = './movielens_data'

    # Training set Preparation
    with open(os.path.join(movielens_dir, 'ua.base'), 'r') as f:
        data = f.readlines()

    train_u = []
    train_i = []
    train_r = []
    for l in data:
        l = re.sub('\n', '', l)
        l = l.split('\t')
        train_u.append(np.int64(l[0])-1)
        train_i.append(np.int64(l[1])-1)
        train_r.append(np.int64(l[2]))

    train = sparse.csr_matrix((train_r, (train_u, train_i)))

    # Test set preparation
    with open(os.path.join(movielens_dir, 'ua.test'), 'r') as f:
            data = f.readlines()

    test_u = []
    test_i = []
    test_r = []
    for l in data:
        l = re.sub('\n', '', l)
        l = l.split('\t')
        test_u.append(np.int64(l[0])-1)
        test_i.append(np.int64(l[1])-1)
        test_r.append(np.int64(l[2]))

    test = sparse.csr_matrix((test_r, (test_u, test_i)))

    # svd = SVD.SVD(train=train, test=test, random_state=89)
    # svd.fit()

    # svd_pp = SVDpp.SVDpp(train=train, test=test, random_state=89)
    # svd_pp.fit()

    svd_integrated = SVD_integrated.SVD_integrated(train=train, test=test, random_state=89)
    svd_integrated.fit()