import time

import utils
import dgi
import logreg

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    random_state = None

    batch_size = 1
    n_epochs = 10000
    patience = 20
    lr = 0.001
    weight_decay = 0.0
    n_hidden = 512
    train_val_test_ratio = [0.8, 0.1, 0.1]

    # Set seed
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)

    # Import data
    adj_mat, features, labels = utils.load_data()

    n_nodes = features.shape[1]
    n_features = features.shape[2]
    n_class = labels.shape[2]

    # train-test split
    idx = list(range(labels.shape[1]))
    train_cut = np.round(len(idx) * train_val_test_ratio[0], 0).astype(int)
    val_cut = np.round(len(idx) * (1 - train_val_test_ratio[2]), 0).astype(int)
    np.random.shuffle(idx)

    train_idx = torch.LongTensor(idx[:train_cut])
    val_idx = torch.LongTensor(idx[train_cut:val_cut])
    test_idx = torch.LongTensor(idx[val_cut:])

    model = dgi.DGI(n_features, n_hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cuda
    model.cuda()
    features = features.cuda()
    adj_mat = adj_mat.cuda()
    labels = labels.cuda()
    train_idx = train_idx.cuda()
    val_idx = val_idx.cuda()
    test_idx = test_idx.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best_score = 1e9
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        corrupted_fts = utils.corrupt_fn(features)

        lbl_1 = torch.ones(batch_size, n_nodes)
        lbl_0 = torch.zeros(batch_size, n_nodes)
        lbl = torch.cat((lbl_1, lbl_0), 1)

        corrupted_fts = corrupted_fts.cuda()
        lbl = lbl.cuda()

        logits = model(features, corrupted_fts, adj_mat)

        loss = b_xent(logits, lbl)

        print('Loss:', loss)

        if loss < best_score:
            best_score = loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stoppping!')
            break

        loss.backward()
        optimizer.step()

    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('best_dgi.pkl'))

    # Get Embedding
    embeds, _ = model.embed(features, adj_mat)
    train_embs = embeds[0, train_idx]
    # val_embs = embeds[0, val_idx]
    test_embs = embeds[0, test_idx]

    train_lbls = torch.argmax(labels[0, train_idx], dim=1)
    # val_lbls = torch.argmax(labels[0, val_idx], dim=1)
    test_lbls = torch.argmax(labels[0, test_idx], dim=1)

    tot_acc = torch.zeros(1).cuda()

    accs = []

    for _ in range(50):
        LogReg = logreg.logreg(n_hidden, n_class)
        optimizer1 = optim.Adam(LogReg.parameters(), lr=0.01, weight_decay=weight_decay)
        LogReg.cuda()

        # pat_steps = 0
        # best_acc = torch.zeros(1).cuda()

        for _ in range(100):
            LogReg.train()
            optimizer1.zero_grad()

            logits = LogReg(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            optimizer1.step()

        logits = LogReg(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot_acc += acc

    print('Average accuracy:', tot_acc / 50)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())




if __name__ == '__main__':
    main()
