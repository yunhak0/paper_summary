import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=200)
    parser.add_argument('--seed', type=int, help='Random Seed', default=None)
    parser.add_argument('--n_infeatures', type=int, help='Number of input features', default=1433)
    parser.add_argument('--n_hidden', type=int, help='Number of hidden features per layer', default=32)
    parser.add_argument('--dim_z', type=int, help='Dimension of latent variables', default=16)
    parser.add_argument('--lr', type=float, help='Adam Optimizer learning rate', default=0.01)

    # Dataset related
    parser.add_argument('--dataset_name', help='Dataset name', default='cora')

    args = parser.parse_args()

    return args
