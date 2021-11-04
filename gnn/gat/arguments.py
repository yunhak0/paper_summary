import argparse

str2bool = lambda x: x.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser()

    # Training Hyperparameter
    parser.add_argument('--n_epochs', type=int, help='number of training epochs.', default=10000)
    parser.add_argument('--patience', type=int, help='number of epochs without improvement on validation before terminating.', default=1000)
    parser.add_argument('--lr', type=float, help='model learning rate.', default=5e-3)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization on model weights.', default=5e-4)
    parser.add_argument('--seed', type=int, help='Random seed.', default=42)
    parser.add_argument('--n_layers', type=int, help='Number of GAT layers.', default=2)
    parser.add_argument('--n_heads', type=int, help='Number of heads per layer.', default=8)
    parser.add_argument('--n_heads_final_layer', type=int, help='Number of heads of attention in output layer', default=1)
    parser.add_argument('--n_hidden', type=int, help='Number of hidden features per layer.', default=8)
    parser.add_argument('--leaky_relu_alpha', type=float, help='Alpha for Leaky ReLU.', default=0.2)
    parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.6)
    parser.add_argument('--add_skip_connection', type=str2bool, help='Skip connection of ResNet', default=False)

    # Dataset related
    parser.add_argument('--dataset_name', help='Dataset for training', default='cora')

    args = parser.parse_args()

    return args

def parse_args_ppi():
    parser = argparse.ArgumentParser()

    # Training Hyperparameter
    parser.add_argument('--n_epochs', type=int, help='number of training epochs.', default=500)
    parser.add_argument('--patience', type=int, help='number of epochs without improvement on validation before terminating.', default=200)
    parser.add_argument('--lr', type=float, help='model learning rate.', default=5e-3)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization on model weights.', default=0)
    parser.add_argument('--seed', type=int, help='Random seed.', default=42)
    parser.add_argument('--n_layers', type=int, help='Number of GAT layers.', default=3)
    parser.add_argument('--n_heads', type=int, help='Number of heads per layer.', default=4)
    parser.add_argument('--n_heads_final_layer', type=int, help='Number of heads of attention in output layer', default=6)
    parser.add_argument('--n_hidden', type=int, help='Number of hidden features per layer.', default=256)
    parser.add_argument('--leaky_relu_alpha', type=float, help='Alpha for Leaky ReLU.', default=0.2)
    parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.0)
    parser.add_argument('--add_skip_connection', type=str2bool, help='Skip connection of ResNet', default=True)

    # Dataset related
    parser.add_argument('--dataset_name', help='Dataset for training', default='ppi')

    # For Inductive Task
    parser.add_argument('--ppi_load_test_only', type=str2bool, help='Load PPI Test dataset only', default=False)
    parser.add_argument('--batch_size', type=int, help='Batch Size.', default=2)

    args = parser.parse_args()

    return args
