from datetime import datetime
import numpy as np
from scipy import sparse
# import matplotlib.pyplot as plt
# import utils

class SVD():
    """SVD

    Implementation of 'Factorization Meets the Neighborhood' by Koren

    Args:
        train (sparse.csr_matrix): training rating matrix with csr_matrix format
        test (sparse.csr_matrix): test rating matrix with csr_matrix format
        epochs (int): the number of epochs for gradient descent. Defaults to 200.
        gamma (float): learning rate or step size of gradient descent.
        Defaults to 0.007.
        lambda3 (float): regularizing term of gradient descent.
        Defaults to 0.005.
        n_factor (int): the number of factors. Defaults to 50.
        random_state (int): RandomState instance or None. Defaults to None.
    """
    def __init__(self,
                 train,
                 test,
                 epochs=200,
                 gamma=0.007,
                 lambda3=0.005,
                 n_factor=50,
                 random_state=None):
        """
        Constructs all the necessary attributes for the SVD object.

        Args:
            train (sparse.csr_matrix): training rating matrix with csr_matrix format
            test (sparse.csr_matrix): test rating matrix with csr_matrix format
            epochs (int): the number of epochs for gradient descent. Defaults to 200.
            gamma (float): learning rate or step size of gradient descent.
            Defaults to 0.007.
            lambda3 (float): regularizing term of gradient descent.
            Defaults to 0.005.
            n_factor (int): the number of factors. Defaults to 50.
            random_state (int): , RandomState instance or None. Defaults to None.
        """
        # Input
        # explicit feedback
        self.R = train
        self.R_test = test

        # Characteristic of training set
        self.mu = train.data.mean()
        self.n_user = train.shape[0]
        self.n_item = train.shape[1]

        # predict rating matrix
        self.R_predicted = sparse.lil_matrix((self.n_user, self.n_item))

        # Input constants
        self.epochs = epochs
        self.gamma = gamma
        self.lambda3 = lambda3
        self.n_factor = n_factor
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize the parameters
        self.b_u = np.random.standard_normal(self.n_user)
        self.b_i = np.random.standard_normal(self.n_item)
        self.q_i = np.random.standard_normal((self.n_item, self.n_factor))
        self.p_u = np.random.standard_normal((self.n_user, self.n_factor))

        # Loss
        self.train_loss = {}
        self.test_loss = {}

    def fit(self):
        """
        Training the SVD model
        """
        start = datetime.now()
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
                    train_error = self.gradient_descent(u, i, r)
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
                r_hat = self.predict(u, i)
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
                print("[Epoch: %d] Training RMSE: %.4f (ETA: %s), Test RMSE: %.4f" %
                        (epoch + 1,
                         self.train_loss[epoch]['RMSE'],
                         eta,
                         self.test_loss[epoch]['RMSE']))
                tmp_timestamp = datetime.now()
        print(f'Processing Time: {datetime.now() - start}')

    def predict(self, u, i):
        """
        Predict the rating of user u and item i

        Args:
            u (int): user u
            i (int): item i

        Returns:
            float or List[float]: predicted rating of user u and item i.
        """
        b_ui = self.mu + self.b_u[u] + self.b_i[i] 
        hat_r_ui = b_ui + np.dot(self.p_u[u], self.q_i[i].T)
        self.R_predicted[u, i] = hat_r_ui
        return hat_r_ui
        
    def gradient(self, u, i, r):
        """
        Calculate the gradient and error of user u and item i

        Args:
            u (int): user u
            i (int): item i
            r (int): rating of user u and item i

        Returns:
            floats: gradients and error of user u and item i.
            d_bu, d_bi, d_qi, d_pu, e_ui
        """
        hat_r_ui = self.predict(u, i)
        e_ui = r - hat_r_ui

        # gradient
        d_bu = e_ui - self.lambda3 * self.b_u[u]
        d_bi = e_ui - self.lambda3 * self.b_i[i]
        d_qi = e_ui * self.p_u[u] - self.lambda3 * self.q_i[i]
        d_pu = e_ui * self.q_i[i] - self.lambda3 * self.p_u[u]

        return d_bu, d_bi, d_qi, d_pu, e_ui

    def gradient_descent(self, u, i, r):
        """
        Apply the gradient descent algorithm to get the parameters

        Args:
            u (int): user u
            i (int): item i
            r (int): rating of user u and item i

        Returns:
            float: prediction error of user u and item i.
            gradient descent is updated in the object itself:
            b_u, b_i, q_i, p_u.

        """
        d_bu, d_bi, d_qi, d_pu, e_ui = self.gradient(u, i, r)

        self.b_u[u] = self.b_u[u] + self.gamma * d_bu
        self.b_i[i] = self.b_i[i] + self.gamma * d_bi
        self.q_i[i] = self.q_i[i] + self.gamma * d_qi
        self.p_u[u] = self.p_u[u] + self.gamma * d_pu

        return e_ui
