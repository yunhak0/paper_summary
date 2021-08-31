from datetime import datetime
import numpy as np
from scipy import sparse


class SVDpp():

    def __init__(self,
                 r_train,
                 r_test,
                 epochs=200,
                 gamma1=0.007,
                 gamma2=0.007,
                 gamma3=0.001,
                 l_reg6=0.005,
                 l_reg7=0.015,
                 l_reg8=0.015,
                 k=300,
                 n_factor=50,
                 random_state=None):
        # Input
        self.r_train = r_train
        self.r_test = r_test
        self.epochs = epochs
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.l_reg6 = l_reg6
        self.l_reg7 = l_reg7
        self.l_reg8 = l_reg8
        self.k = k
        self.n_factor = n_factor
        if random_state is not None:
            np.random.seed(random_state)

        # Characteristic of training set
        self.mu = r_train.data.mean()
        self.n_user = r_train.shape[0]
        self.n_item = r_train.shape[1]

        # Initialize the parameter
        self.b_u = np.random.rand(self.n_user, 1)
        self.b_i = np.random.rand(1, self.n_item)
        self.w_ij = np.random.rand(self.n_item, self.n_item)
        self.c_ij = np.random.rand(self.n_item, self.n_item)
        self.q_i = np.random.rand(self.n_item, n_factor)
        self.p_u = np.random.rand(self.n_user, n_factor)
        self.y_j = np.random.rand(self.n_item, n_factor)

        def get_index(self, r_train):
            dict_idx = {'data_idx': [], 'user': [], 'item':[]}
            for u, (begin, end) in enumerate(zip(r_train.indptr,
                                                 r_train.indptr[1:])):
                dict_idx['user'].append(u)
                dict_idx['data_idx'].append(list(range(begin, end)))
                dict_idx['item'].append(list(r_train.indices[begin:end]))
            return dict_idx
        
        def fit(self):
            start = datetime.now()
            dict_idx = get_index(self.r_train)
            r_train_coo = self.r_train.tocoo()
            for epoch in range(epochs):
                for u, i, r in zip(r_train_coo.row, r_train_coo.col, r_train_coo.data):
                    N_u = dict_idx['item'][u]
                    # In this Netflix data, there are some limitaion of implicit feeback
                    # The implicit feedback is defined by the binary matrix,
                    # where '1' stands for 'rated', and '0' for 'not rated'.
                    # Therefore, R_k_iu = N_k_iu
                    # R_k_iu: all items for which ratings by u are available
                    # N_k_iu: all the items for which u provided an implicit feedback.
                    R_k_iu = 
                    N_k_iu = 
