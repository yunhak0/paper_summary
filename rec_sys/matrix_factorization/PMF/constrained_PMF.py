from datetime import datetime
from tqdm import tqdm
import numpy as np
from scipy import sparse

class constrained_PMF():
    """Constrained Probabilistic Matrix Factorization Model

    Implementation of 'Probabilistic Matrix Factorization'
    by R. Salakhutdinov and A. Mnih

    Args:
        train (sparse.csr_matrix): training rating matrix with csr_matrix format.
        test (sparse.csr_matrix): test rating matrix with csr_matrix format.
        scale_r (bool): whether rating is applied to the scale function (t(x))
        D (int, optional): the number of factors. Defaults to 10.
        sigma (float, optional): the variance of Gaussian distribution
        for the probability density function of p(R|U,V,sigma^2).
        Defaults to 0.01.
        sigma_U (float, optional): the variance of Gaussian distribution
        for the probability density function of p(U|0,sigma_U^2) - user matrix.
        Defaults to 0.1.
        sigma_V (float, optional): the variance of Gaussian distribution
        for the probability density function of p(V|0,sigma_V^2) - item matrix.
        Defaults to 0.1.
        epochs (int, optional): the number of epochs for gradient descent.
        Defaults to 200.
        learning_rate (float, optional): learning rate or step size of gradient descent.
        Defaults to 0.005.
        random_state (None or int, optional): Random State instance or None.
        Defaults to None.
    """
    def __init__(self,
                 train,
                 test,
                 scale_r=True,
                 D=30,
                 sigma=0.01,
                 sigma_Y=0.1,
                 sigma_W=0.1,
                 sigma_V=0.1,
                 epochs=100,
                 learning_rate=0.005,
                 random_state=None):
        """Probabilistic Matrix Factorization Model

        Args:
            train (sparse.csr_matrix): training rating matrix with csr_matrix format.
            test (sparse.csr_matrix): test rating matrix with csr_matrix format.
            scale_r (bool): whether rating is applied to the scale function (t(x)).
            D (int, optional): the number of factors. Defaults to 10.
            sigma (float, optional): the variance of Gaussian distribution
            for the probability density function - p(R|Y,V,W,sigma^2).
            Defaults to 0.01.
            sigma_Y (float, optional): the variance of Gaussian distribution
            for the probability density function of p(Y|sigma_Y^2) - user matrix.
            Defaults to 0.1.
            sigma_W (float, optional): the variance of Gaussian distribution
            for the probability density function of p(W|sigma_W^2)
            - offset of user matrix.
            Defaults to 0.1.
            sigma_V (float, optional): the variance of Gaussian distribution
            for the probability density function of p(V|sigma_V^2) - item matrix.
            Defaults to 0.1.
            epochs (int, optional): the number of epochs for gradient descent.
            Defaults to 200.
            learning_rate (float, optional): learning rate or step size of gradient descent.
            Defaults to 0.005.
            random_state (None or int, optional): Random State instance or None.
            Defaults to None.
        """        
        self.R = train
        self.R_test = test
        self.scale_r = scale_r
        self.D = D
        self.sigma = sigma
        self.sigma_Y = sigma_Y
        self.sigma_W = sigma_W
        self.sigma_V = sigma_V
        self.epochs = epochs
        self.learning_rate = learning_rate
        if random_state is not None:
            np.random.seed(random_state)

        self.K = np.max(train)
        self.N, self.M = train.shape
        # implicit feedback
        nnz_idx = train.nonzero()
        keep = np.where(train.data != 0)[0]
        n_keep = len(np.where(train.data != 0)[0])
        self.I = sparse.csr_matrix((np.ones(n_keep), (nnz_idx[0][keep], nnz_idx[1][keep])))

        self.R_predicted = sparse.lil_matrix((self.N, self.M))

        self.train_loss = {}
        self.test_loss = {}

        # Initialize
        ## offset added to the mean of the prior dist. to get the U_i for the user i
        self.Y = np.random.normal(loc=0.0, scale=self.sigma_V, size=(self.N, self.D))
        ## latent similarity constraint matrix
        self.W = np.random.normal(loc=0.0, scale=self.sigma_V, size=(self.M, self.D))
        self.V = np.random.normal(loc=0.0, scale=self.sigma_V, size=(self.M, self.D))

        self.lambda_Y = (self.sigma / self.sigma_Y)**2
        self.lambda_W = (self.sigma / self.sigma_W)**2
        self.lambda_V = (self.sigma / self.sigma_V)**2

    def fit(self):
        """
        Training the PMF model
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
            
            for i, j, r in zip(R_coo.row,
                               R_coo.col,
                               R_coo.data):
                if r > 0:
                    if self.scale_r:
                        r = self.t(r)
                    train_error = self.gradient_descent(i, j, r)
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
            for i, j, r in zip(R_test_coo.row,
                               R_test_coo.col,
                               R_test_coo.data):
                if r > 0:
                    r_hat = self.predict(i, j)
                    if self.scale_r:
                        r = self.t(r)
                        r_hat = self.g(r_hat)
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

    def predict(self, i, j):
        # predict
        user_matrix = self.Y[i] + np.dot(self.I[i].todense(), self.W)/self.I[i].sum()
        item_matrix = self.V[j]
        r_hat_ij = np.dot(user_matrix, item_matrix).item()
        self.R_predicted[i, j] = r_hat_ij
        return r_hat_ij

    def g(self, x):
            """Logistic Function

            It bounds the range of predictions.

            Args:
                x (float): Predicted Rating (r_hat_ij)
            
            Returns:
                float: Scaled predicted rating using logistic function
            """
            return 1/(1 + np.exp(-x))

    def t(self, x):
        """Scale Fucntion

        The range of valid rating values matches the range of predictions our model makes.

        Args:
            x (float): Predicted Rating (r_hat_ij) after logistic function is applied (g(x))

        Returns:
            float: Scaled predicted rating
        """
        return (x - 1) / (self.K - 1)

    def gradient(self, i, j, r):

        if self.scale_r:
            r_hat_ij = self.g(self.predict(i, j))
        
        e_ij = r - r_hat_ij

        # gradient
        if self.scale_r:
            d_Y = e_ij * (r_hat_ij * (1 - r_hat_ij)) * self.V[j] - self.lambda_Y * self.Y[i]
            d_W = e_ij * (r_hat_ij * (1 - r_hat_ij)) \
                  * (np.outer(self.I[i].todense() / self.I[i].sum(), self.V[j])) \
                  - self.lambda_W * self.W[j]
            d_V = e_ij * (r_hat_ij * (1 - r_hat_ij)) \
                  * (self.Y[i] + self.I[i] * self.W / self.I[i].sum()) \
                  - self.lambda_V * self.V[j]
        else:
            d_Y = e_ij * self.V[j] - self.lambda_Y * self.Y[i]
            d_W = e_ij * (np.outer(self.I[i].todense() / self.I[i].sum(), self.V[j])) - self.lambda_W * self.W[j]
            d_V = e_ij * (self.Y[i] + self.I[i] * self.W / self.I[i].sum()) - self.lambda_V * self.V[j]

        return d_Y, d_W, d_V, e_ij

    def gradient_descent(self, i, j, r):
        d_Y, d_W, d_V, e_ij = self.gradient(i, j, r)

        # gradient descent
        self.Y[i] = self.Y[i] + self.learning_rate * d_Y
        self.W = self.W + self.learning_rate * d_W
        self.V[j] = self.V[j] + self.learning_rate * d_V

        return e_ij
