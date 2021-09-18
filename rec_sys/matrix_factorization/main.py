import os
import re
import numpy as np
from scipy import sparse
from SVD_integrated import SVD, Asymmetric_SVD, SVDpp, SVD_integrated
from PMF import PMF, constrained_PMF

if __name__ == '__main__':
    movielens_dir = './movielens_data'

    # Training set Preparation
    with open(os.path.join(movielens_dir, 'u1.base'), 'r') as f:
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
    with open(os.path.join(movielens_dir, 'u1.test'), 'r') as f:
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

    # print('SVD -----------------------------------------------')
    # svd = SVD.SVD(train=train, test=test, random_state=89)
    # svd.fit()

    # print('SVD++ ---------------------------------------------')
    # svd_pp = SVDpp.SVDpp(train=train, test=test, random_state=89)
    # svd_pp.fit()

    # print('SVD Integrated ------------------------------------')
    # svd_integrated = SVD_integrated.SVD_integrated(train=train, test=test, random_state=89)
    # svd_integrated.fit()

    print('PMF -----------------------------------------------')
    pmf = PMF.PMF(train=train, test=test, random_state=89)
    pmf.fit()

    print('Constrained PMF -------------------------------------')
    constrained_pmf = constrained_PMF.constrained_PMF(train=train, test=test, random_state=89)
    constrained_pmf.fit()
