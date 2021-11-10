import time
import numpy as np
import utils
from arguments import parse_args

import scipy.sparse as sp

import torch
from torch import optim

import vgae

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    my_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    adj, features = utils.load_data(args.dataset_name)
    n_nodes, in_features = features.shape

    # Preprocessing - adjacency matrix without diagonal entries
    adj_org = adj
    adj_org = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
    adj_org.eliminate_zeros()

    # Edge masking & Split training, validation, and test
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = utils.mask_test_edges(adj)
    adj = adj_train

    # Preprocessing for loss
    adj_norm = utils.preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).to(my_dev)
    norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)).to(my_dev)

    # To Device
    adj_norm = torch.FloatTensor(adj_norm.todense()).to(my_dev)
    adj_label = torch.FloatTensor(adj_label.todense()).to(my_dev)
    features = torch.FloatTensor(features.todense()).to(my_dev)

    gae_model = vgae.gae(in_features, args.n_hidden, args.dim_z).to(my_dev)
    vgae_model = vgae.vgae(in_features, args.n_hidden, args.dim_z).to(my_dev)

    opt_gae = optim.Adam(gae_model.parameters(), lr=args.lr)
    opt_vgae = optim.Adam(vgae_model.parameters(), lr=args.lr)

    BEST_GAE_VAL_SCORE = 0
    BEST_GAE_VAL_LOSS = 0
    PATIENCE_GAE_CNT = 0

    BEST_VGAE_VAL_SCORE = 0
    BEST_VGAE_VAL_LOSS = 0
    PATIENCE_VGAE_CNT = 0

    # GAE Training
    print('GAE TRAINING -----')
    start = time.time()
    for epoch in range(args.n_epochs):
        t = time.time()

        #gae_model.train()
        opt_gae.zero_grad()
        gae_outs = gae_model(features, adj_norm)
        gae_loss = gae_model.loss(gae_outs, adj_label, pos_weight, norm)
        gae_loss.backward()
        opt_gae.step()

        train_acc = utils.get_train_acc(gae_outs, adj_label)
        val_roc, val_avg_prec = utils.get_scores(gae_outs, adj_label, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(gae_loss.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_avg_prec),
              "time=", "{:.5f}".format(time.time() - t))

        if val_roc > BEST_GAE_VAL_SCORE or gae_loss.item() < BEST_GAE_VAL_LOSS:
            BEST_GAE_VAL_SCORE = max(val_roc, BEST_GAE_VAL_SCORE)
            BEST_GAE_VAL_LOSS = min(gae_loss.item(), BEST_GAE_VAL_LOSS)
            PATIENCE_GAE_CNT = 0
        else:
            PATIENCE_GAE_CNT += 1
        
        if PATIENCE_GAE_CNT > 100:
            print('Stopping the training due to patience setting')
            break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - start))
    print('BEST GAE VALIDATION SCORE',
            '| Best Val Loss: {:.4f}'.format(BEST_GAE_VAL_LOSS),
            '| Best Val ROC AUC: {:.4f}'.format(BEST_GAE_VAL_SCORE))

    test_roc, test_avg_prec = utils.get_scores(gae_outs, adj_org, test_edges, test_edges_false)
    print('GAE TEST SCORE',
          '| Test Loss: {:.4f}'.format(test_roc),
          '| Test ROC AUC: {:.4f}'.format(test_avg_prec))


if __name__ == '__main__':
    main()
