import time
from torch._C import device
import utils
import numpy as np
from arguments import parse_args
import torch
import torch.nn as nn
from torch.optim import Adam

import gat

def main():
    args = parse_args()

    # Set Up Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Device
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data(args.dataset_name)

    adj = adj.to(my_device)
    features = features.to(my_device)
    labels = labels.to(my_device)
    idx_train = idx_train.to(my_device)
    idx_val = idx_val.to(my_device)
    idx_test = idx_test.to(my_device)

    my_gat = gat.GAT(
        n_layers=args.n_layers,
        n_feature=features.shape[1],
        n_hidden=args.n_hidden,
        n_class=len(labels.unique()),
        n_heads=[args.n_heads, args.n_heads_final_layer],
        leaky_relu_alpha=args.leaky_relu_alpha,
        dropout=args.dropout,
        activation=nn.ELU(),
        skip_connection=args.add_skip_connection,
        concat=True
    ).to(my_device)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(my_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lbl_train = labels.index_select(0, idx_train)
    lbl_val = labels.index_select(0, idx_val)
    lbl_test = labels.index_select(0, idx_test)

    graph_data = (features, adj)

    BEST_VAL_ACC = 0
    BEST_VAL_LOSS = 0
    PATIENCE_CNT = 0

    # Training
    start = time.time()
    for epoch in range(args.n_epochs):
        my_gat.train()
        output = my_gat(graph_data)[0]
        unnorm_node_class_scores = output.index_select(0, idx_train)
        loss = loss_fn(unnorm_node_class_scores, lbl_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_class = torch.argmax(unnorm_node_class_scores, dim=-1)
        acc = torch.sum(torch.eq(pred_class, lbl_train).long()).item() / len(lbl_train)

        with torch.no_grad():
            my_gat.eval()
            output_val = my_gat(graph_data)[0]
            unnorm_node_class_scores_val = output_val.index_select(0, idx_val)
            loss_val = loss_fn(unnorm_node_class_scores_val, lbl_val)
            pred_class_val = torch.argmax(unnorm_node_class_scores_val, dim=-1)
            acc_val = torch.sum(torch.eq(pred_class_val, lbl_val).long()).item() / len(lbl_val)

        if epoch % 10 == 0 or epoch == args.n_epochs -1:
            print('Epoch: {:04d}'.format(epoch+1),
                  '| Training Loss: {:.4f}'.format(loss.item()),
                  '| Training Acc: {:.4f}'.format(acc),
                  '| Validation Loss: {:.4f}'.format(loss.item()),
                  '| Validation Acc: {:.4f}'.format(acc_val))

        if acc_val > BEST_VAL_ACC or loss_val.item() < BEST_VAL_LOSS:
            BEST_VAL_ACC = max(acc_val, BEST_VAL_ACC)
            BEST_VAL_LOSS = min(loss_val.item(), BEST_VAL_LOSS)
            PATIENCE_CNT = 0
        else:
            PATIENCE_CNT += 1
        
        if PATIENCE_CNT > args.patience:
            print('Stopping the training due to patience setting')
            break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - start))

    # Test
    my_gat.eval()
    output_test = my_gat(graph_data)[0]
    unnorm_node_class_scores_test = output_test.index_select(0, idx_test)
    loss_test = loss_fn(unnorm_node_class_scores_test, lbl_test)
    pred_class_test = torch.argmax(unnorm_node_class_scores_test, dim=-1)
    acc_test = torch.sum(torch.eq(pred_class_test, lbl_test).long()).item() / len(lbl_test)

    print('Test Results ----\n',
          'Test Loss: {:.4f}'.format(loss_test.item()),
          'Test Acc: {:.4f}'.format(acc_test))

    torch.save(
        utils.get_model_state(args, my_gat),
        './gnn/gat/best_model/'+ args.dataset_name + str(int(time.time())) + '.pth'
    )


if __name__ == '__main__':
    main()
