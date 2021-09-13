from datetime import datetime
from tqdm import tqdm
import numpy as np
from scipy import sparse
from SVD_integrated import utils

class SVD_integrated():
    """SVD_integrated

    Implementation of 'Factorization Meets the Neighborhood' by Koren

    Args:
        train (sparse.csr_matrix): training rating matrix with csr_matrix format
        test (sparse.csr_matrix): test rating matrix with csr_matrix format
        epochs (int): the number of epochs for gradient descent. Defaults to 200.
        gamma1 (float): learning rate or step size of gradient descent for baseline.
        Defaults to 0.007.
        gamma2 (float): learning rate or step size of gradient descent
        for user, item matrix and implicit feedback. Defaults to 0.007.
        gamma3 (float): learning rate or step size of gradient descent
        for neighborhood model. Defaults to 0.001.
        lambda6 (float): regularizing term of gradient descent for baseline.
        Defaults to 0.005.
        lambda7 (float): regularizing term of gradient descent
        for user, item matrix and implicit feedback. Defaults to 0.015.
        lambda8 (float): regularizing term of gradient descent
        for neighborhood model. Defaults to 0.015.
        k (int): the number of similar items (neighbors) k.
        lambda2 (int): regularizing term of similarity
        for neighborhood model. Defaults to 100.
        n_factor (int): the number of factors. Defaults to 50.
        random_state (int): RandomState instance or None. Defaults to None.
    """
    def __init__(self,
                 train,
                 test,
                 epochs=200,
                 gamma1=0.007,
                 gamma2=0.007,
                 gamma3=0.001,
                 lambda6=0.005,
                 lambda7=0.015,
                 lambda8=0.015,
                 k=300,
                 lambda2=100,
                 n_factor=50,
                 random_state=None):
        """SVD_integrated

        Implementation of 'Factorization Meets the Neighborhood' by Koren

        Args:
            train (sparse.csr_matrix): training rating matrix with csr_matrix format
            test (sparse.csr_matrix): test rating matrix with csr_matrix format
            epochs (int): the number of epochs for gradient descent. Defaults to 200.
            gamma1 (float): learning rate or step size of gradient descent for baseline.
            Defaults to 0.007.
            gamma2 (float): learning rate or step size of gradient descent
            for user, item matrix and implicit feedback. Defaults to 0.007.
            gamma3 (float): learning rate or step size of gradient descent
            for neighborhood model. Defaults to 0.001.
            lambda6 (float): regularizing term of gradient descent for baseline.
            Defaults to 0.005.
            lambda7 (float): regularizing term of gradient descent
            for user, item matrix and implicit feedback. Defaults to 0.015.
            lambda8 (float): regularizing term of gradient descent
            for neighborhood model. Defaults to 0.015.
            k (int): the number of similar items (neighbors) k.
            lambda2 (int): regularizing term of similarity
            for neighborhood model. Defaults to 100.
            n_factor (int): the number of factors. Defaults to 50.
            random_state (int): RandomState instance or None. Defaults to None.
        """
        # Input
        # explicit feedback
        self.R = train
        self.R_test = test

        # implicit feedback
        self._nnz_idx = train.nonzero()
        self._keep = np.where(train.data != 0)[0]
        self._n_keep = len(self._keep)
        self.N = sparse.csr_matrix((np.ones(self._n_keep),
                                    (self._nnz_idx[0][self._keep],
                                     self._nnz_idx[1][self._keep])))

        # Characteristic of training set
        self.mu = train.data.mean()
        self.n_user = train.shape[0]
        self.n_item = train.shape[1]

        # predict rating matrix
        self.R_predicted = sparse.lil_matrix((self.n_user, self.n_item))

        # Input constants
        self.epochs = epochs
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.lambda6 = lambda6
        self.lambda7 = lambda7
        self.lambda8 = lambda8
        self.n_factor = n_factor
        self.k = k
        self.lambda2 = lambda2
        self.n_factor = n_factor
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize the parameters
        self.b_u = np.random.standard_normal(self.n_user)
        self.b_i = np.random.standard_normal(self.n_item)
        self.q_i = np.random.standard_normal((self.n_item, self.n_factor))
        self.p_u = np.random.standard_normal((self.n_user, self.n_factor))
        self.y_j = np.random.standard_normal((self.n_item, self.n_factor))
        self.w_ij = np.random.standard_normal((self.n_item, self.n_item))
        self.c_ij = np.random.standard_normal((self.n_item, self.n_item))

        # Loss
        self.train_loss = {}
        self.test_loss = {}

        # Similarity
        self.sk_i = np.zeros((self.n_item, self.k))

    def get_similar(self):
        rho = utils.sparse_corrcoef(self.R)
        n_ij = np.dot(self.N.T, self.N)
        rows, cols = n_ij.nonzero()
        s_ij = sparse.lil_matrix((self.n_item, self.n_item))
        for i, j in tqdm(zip(rows, cols),
                         total=len(rows),
                         desc='Get s_ij'):
            s_ij[i, j] = n_ij[i, j] / (n_ij[i, j] + self.lambda2) * rho[i, j]
        
        for i in tqdm(range(self.n_item),
                      desc='Get sk_i'):
            s_i = s_ij[i].todense()
            self.sk_i[i] = s_i.argsort().tolist()[0][-self.k:]

    def fit(self):
        """
        Training the SVD_integrated model
        """
        start = datetime.now()

        # Similarity
        self.get_similar()
        
        R_coo = self.R.tocoo()
        R_test_coo = self.R_test.tocoo()
        for epoch in range(self.epochs):
            epoch_start = datetime.now()

            # Training -------------------------------------------
            self.train_loss[epoch] = {}
            self.train_loss[epoch]['error'] = []
            training_check_point = 0
            
            for u, i, r in zip(R_coo.row,
                               R_coo.col,
                               R_coo.data):
                if r > 0:
                    R_u = self.R[u].indices
                    N_u = self.N[u].indices
                    Rk_iu = [int(v) for v in np.intersect1d(R_u, self.sk_i[i])]
                    Nk_iu = [int(v) for v in np.intersect1d(N_u, self.sk_i[i])]
                    train_error = self.gradient_descent(u, i, r, N_u, Rk_iu, Nk_iu)
                    self.train_loss[epoch]['error'].append(train_error)
                    training_check_point += 1

            train_rmse = np.sqrt(
                np.sum([e ** 2 for e in self.train_loss[epoch]['error']]) /
                len(self.train_loss[epoch]['error'])
            )
            self.train_loss[epoch]['RMSE'] = train_rmse
            self.train_loss[epoch]['ETA'] = datetime.now() - epoch_start

            # Test --------------------------------------------
            self.test_loss[epoch] = {}
            self.test_loss[epoch]['error'] = []

            # Create Completed Predicted Rating Matrix
            test_check_point = 0
            for u, i, r in zip(R_test_coo.row,
                               R_test_coo.col,
                               R_test_coo.data):
                if r > 0:
                    R_u = self.R[u].indices
                    N_u = self.N[u].indices
                    Rk_iu = [int(v) for v in np.intersect1d(R_u, self.sk_i[i])]
                    Nk_iu = [int(v) for v in np.intersect1d(N_u, self.sk_i[i])]
                    r_hat = self.predict(u, i, N_u, Rk_iu, Nk_iu)
                    test_error = r - r_hat
                    self.test_loss[epoch]['error'].append(test_error)
                    test_check_point += 1

            test_rmse = np.sqrt(
                np.sum([e ** 2 for e in self.test_loss[epoch]['error']]) /
                len(self.test_loss[epoch]['error'])
            )
            self.test_loss[epoch]['RMSE'] = test_rmse

            # Progress Info
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                if epoch == 0:
                    eta = (datetime.now() - start).total_seconds()
                else:
                    eta = (datetime.now() - tmp_timestamp).total_seconds()
                eta = divmod(eta, 60)
                eta = str(int(eta[0])) + 'm ' + str(np.round(eta[1], 4)) + 's'
                print("[Epoch: %s] Training RMSE: %.4f (ETA: %s), Test RMSE: %.4f" %
                        (str(epoch + 1).zfill(len(str(self.epochs))),
                         self.train_loss[epoch]['RMSE'],
                         eta,
                         self.test_loss[epoch]['RMSE']))
                tmp_timestamp = datetime.now()
        print(f'Processing Time: {datetime.now() - start}')

    def predict(self, u, i, N_u, Rk_iu, Nk_iu):
        """
        Predict the rating of user u and item i

        Args:
            u (int): user u
            i (int): item i
            N_u (List[int]): All items for which user u provided an implicit feedback.
            Rk_iu (List[int]): Similar k items of explicit feedback by user u
            Nk_iu (List[int]): Similar k items of implicit feedback by user u

        Returns:
            float or List[float]: predicted rating of user u and item i.
        """
        # first tier
        tier1 = self.mu + self.b_u[u] + self.b_i[i]
        # second tier
        q_i_T = self.q_i[i].T
        p_u_ = self.p_u[u] + np.sum(self.y_j[N_u], axis=0)/np.sqrt(len(N_u))
        tier2 = np.dot(p_u_, q_i_T)
        # third tier
        if len(Rk_iu) > 0:
            b_uj = self.mu + self.b_u[u] + self.b_i[Rk_iu]
            global_optim_w = np.dot((self.R[u, Rk_iu] - b_uj), self.w_ij[i, Rk_iu]).item()
            global_optim_w = global_optim_w/np.sqrt(len(Rk_iu))
            
        else:
            global_optim_w = 0
        
        if len(Nk_iu) > 0:
            offsets_baseline = self.c_ij[i, Nk_iu].sum()
            offsets_baseline = offsets_baseline/np.sqrt(len(Nk_iu))
        else:
            offsets_baseline = 0
        tier3 = global_optim_w + offsets_baseline
        # predict
        hat_r_ui = tier1 + tier2 + tier3
        self.R_predicted[u, i] = hat_r_ui
        return hat_r_ui
        
    def gradient(self, u, i, r, N_u, Rk_iu, Nk_iu):
        """
        Calculate the gradient and error of user u and item i

        Args:
            u (int): user u
            i (int): item i
            r (int): rating of user u and item i
            N_u (List[int]): All items for which user u provided an implicit feedback.
            Rk_iu (List[int]): Similar k items of explicit feedback by user u
            Nk_iu (List[int]): Similar k items of implicit feedback by user u

        Returns:
            floats: gradients and error of user u and item i.
            d_bu, d_bi, d_qi, d_pu, d_yj, d_wij, d_cij, e_ui
        """
        hat_r_ui = self.predict(u, i, N_u, Rk_iu, Nk_iu)
        e_ui = r - hat_r_ui

        # gradient
        d_pu = e_ui * self.q_i[i] - self.lambda7 * self.p_u[u]
        d_qi = (e_ui * (self.p_u[u] + np.sum(self.y_j[N_u], axis=0) /
                        np.sqrt(len(N_u)))
                - self.lambda7 * self.q_i[i])

        d_yj = e_ui * self.q_i[i]/np.sqrt(len(N_u)) - self.lambda7 * self.y_j[N_u]

        b_uj = self.mu + self.b_u[u] + self.b_i[Rk_iu]
        d_wij = (e_ui * (self.R[u, Rk_iu] - b_uj) /
                         np.sqrt(len(Rk_iu))
                 - self.lambda8 * self.w_ij[i, Rk_iu])
        d_cij = e_ui/np.sqrt(len(Nk_iu)) - self.lambda8 * self.c_ij[i, Nk_iu]

        d_bu = e_ui - self.lambda6 * self.b_u[u]
        d_bi = e_ui - self.lambda6 * self.b_i[i]

        return d_bu, d_bi, d_qi, d_pu, d_yj, d_wij, d_cij, e_ui

    def gradient_descent(self, u, i, r, N_u, Rk_iu, Nk_iu):
        """
        Apply the gradient descent algorithm to get the parameters

        Args:
            u (int): user u
            i (int): item i
            r (int): rating of user u and item i
            N_u (List[int]): All items for which user u provided an implicit feedback.
            Rk_iu (List[int]): Similar k items of explicit feedback by user u
            Nk_iu (List[int]): Similar k items of implicit feedback by user u

        Returns:
            float: prediction error of user u and item i.
            gradient descent is updated in the object itself:
            b_u, b_i, q_i, p_u, y_j, w_ij, c_ij.

        """
        d_bu, d_bi, d_qi, d_pu, d_yj, d_wij, d_cij, e_ui = self.gradient(
            u, i, r, N_u, Rk_iu, Nk_iu
        )

        self.p_u[u] = self.p_u[u] + self.gamma2 * d_pu
        self.q_i[i] = self.q_i[i] + self.gamma2 * d_qi

        self.y_j[N_u] = self.y_j[N_u] + self.gamma2 * d_yj

        if len(Rk_iu) > 0:
            self.w_ij[i, Rk_iu] = self.w_ij[i, Rk_iu] + self.gamma3 * d_wij
        if len(Nk_iu) > 0:
            self.c_ij[i, Nk_iu] = self.c_ij[i, Nk_iu] + self.gamma3 * d_cij

        self.b_u[u] = self.b_u[u] + self.gamma1 * d_bu
        self.b_i[i] = self.b_i[i] + self.gamma1 * d_bi

        return e_ui
