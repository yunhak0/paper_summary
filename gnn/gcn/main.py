import time

import utils
import gcn

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

def main():
    random_state = 42
    n_hidden = 16
    lr = 0.01
    weight_decay = 5e-4
    n_epoch = 200
    dropout = 0.5
    train_val_test_ratio = [0.8, 0.1, 0.1]

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    adj_mat, features, labels = utils.load_data()

    model = gcn.GCN(n_features=features.shape[1],
                    n_hidden=n_hidden,
                    n_class=labels.max().item()+1,
                    dropout=dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    idx = list(range(labels.shape[0]))
    train_cut = np.round(len(idx) * train_val_test_ratio[0], 0).astype(int)
    val_cut = np.round(len(idx) * (1 - train_val_test_ratio[2]), 0).astype(int)
    np.random.shuffle(idx)

    train_idx = torch.LongTensor(idx[:train_cut])
    val_idx = torch.LongTensor(idx[train_cut:val_cut])
    test_idx = torch.LongTensor(idx[val_cut:])

    # cuda
    model.cuda()
    features = features.cuda()
    adj_mat = adj_mat.cuda()
    labels = labels.cuda()
    train_idx = train_idx.cuda()
    val_idx = val_idx.cuda()
    test_idx = test_idx.cuda()


    # Training
    def train(epoch):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_mat)
        loss_train = F.nll_loss(output[train_idx], labels[train_idx])
        acc_train = utils.accuracy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(features, adj_mat)

        loss_val = F.nll_loss(output[val_idx], labels[val_idx])
        acc_val = utils.accuracy(output[val_idx], labels[val_idx])
        if epoch % 10 == 0 or epoch == n_epoch - 1:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}'.format(time.time() - start))

    def test():
        model.eval()
        output = model(features, adj_mat)
        loss_test = F.nll_loss(output[test_idx], labels[test_idx])
        acc_test = utils.accuracy(output[test_idx], labels[test_idx])
        print('Test set results:',
              'loss: {:.4f}'.format(loss_test.item()),
              'accuracy: {:.4f}'.format(acc_test.item()))

    total_time = time.time()
    for epoch in range(n_epoch):
        train(epoch)
    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - total_time))

    test()


if __name__ == '__main__':
    main()
