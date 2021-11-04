import time
from torch._C import device
import utils
import numpy as np
from arguments import parse_args_ppi
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.metrics import f1_score

import gat

def main():
    args = parse_args_ppi()

    # Set Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # Device
    # my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_device = 'cpu'

    # Load the data
    data_loader_train, data_loader_val, data_loader_test = utils.load_ppi_data(args)

    my_gat = gat.GAT(
        n_layers=args.n_layers,
        n_features=50, # data_loader_train.dataset.node_features_list[0].shape
        n_hiddens=args.n_hidden,
        n_class=121, # data_loader_train.dataset.node_labels_list[0].shape
        n_attn_heads=[args.n_heads, args.n_heads_final_layer],
        leaky_relu_alpha=args.leaky_relu_alpha,
        dropout=args.dropout,
        activation=nn.ELU(),
        skip_connection=args.add_skip_connection,
        concat=True
    ).to(my_device)

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(my_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    BEST_VAL_ACC = 0
    BEST_VAL_LOSS = 0
    PATIENCE_CNT = 0

    # Training
    start = time.time()
    TRAINING_CNT = 0
    VALIDATION_CNT = 0
    for epoch in range(args.n_epochs):
        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader_train):
            edge_index = to_scipy_sparse_matrix(edge_index).todense()
            edge_index = torch.Tensor(edge_index).to(my_device)
            node_features = node_features.to(my_device)
            gt_node_labels = gt_node_labels.to(my_device)

            graph_data = (node_features, edge_index)

            unnorm_scores = my_gat(graph_data)[0]

            loss = loss_fn(unnorm_scores, gt_node_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = (unnorm_scores > 0).float().cpu().numpy()
            gt = gt_node_labels.cpu().numpy()
            micro_f1 = f1_score(gt, pred, average='micro')

            TRAINING_CNT += 1
            if TRAINING_CNT % 10 == 0 or TRAINING_CNT == len(data_loader_train) * args.n_epochs + batch_idx:
                print('Epoch: {:04d}'.format(epoch+1),
                    '| Training Count: {:04d}'.format(TRAINING_CNT),
                    '| Training Loss: {:.4f}'.format(loss.item()),
                    '| Training Micro-F1: {:.4f}'.format(micro_f1))
            
        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader_val):
            edge_index = to_scipy_sparse_matrix(edge_index).todense()
            edge_index = torch.Tensor(edge_index).to(my_device)
            node_features = node_features.to(my_device)
            gt_node_labels = gt_node_labels.to(my_device)

            graph_data = (node_features, edge_index)

            unnorm_scores = my_gat(graph_data)[0]

            loss = loss_fn(unnorm_scores, gt_node_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = (unnorm_scores > 0).float().cpu().numpy()
            gt = gt_node_labels.cpu().numpy()
            micro_f1 = f1_score(gt, pred, average='micro')

            VALIDATION_CNT += 1
            if VALIDATION_CNT % 10 == 0 or VALIDATION_CNT == len(data_loader_val) * args.n_epochs + batch_idx:
                print('Epoch: {:04d}'.format(epoch+1),
                    '| Validation Count: {:04d}'.format(VALIDATION_CNT),
                    '| Validation Loss: {:.4f}'.format(loss.item()),
                    '| Validation Micro-F1: {:.4f}'.format(micro_f1))

            if micro_f1 > BEST_VAL_ACC or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_ACC = max(micro_f1, BEST_VAL_ACC)
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                PATIENCE_CNT = 0
                print('VALIDATION_CNT: {:04d} | Updated: F1 - {:.4f}'.format(VALIDATION_CNT, BEST_VAL_ACC))
            else:
                PATIENCE_CNT += 1
            
            if PATIENCE_CNT > args.patience:
                print('Stopping the training due to patience setting')
                break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - start))

    # Test
    for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader_test):
        edge_index = to_scipy_sparse_matrix(edge_index).todense()
        edge_index = torch.Tensor(edge_index).to(my_device)
        node_features = node_features.to(my_device)
        gt_node_labels = gt_node_labels.to(my_device)

        graph_data = (node_features, edge_index)

        unnorm_scores = my_gat(graph_data)[0]

        loss = loss_fn(unnorm_scores, gt_node_labels)

        pred = (unnorm_scores > 0).float().cpu().numpy()
        gt = gt_node_labels.cpu().numpy()
        micro_f1 = f1_score(gt, pred, average='micro')

        print('Test Results ----\n',
              'Test Loss: {:.4f}'.format(loss.item()),
              'Test Micro-F1: {:.4f}'.format(micro_f1))

        torch.save(
            utils.get_model_state(args, my_gat),
            './gnn/gat/best_model/'+ args.dataset_name + str(int(time.time())) + '.pth'
        )





if __name__ == '__main__':
    main()
