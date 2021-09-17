from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import utils
from Graph import Graph
from Tree import Tree
from deepwalk import deepwalk

from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == '__main__':
    data_dir = './gnn/data/social_data'
    file_name = 'blogcatalog.mat'

    # Parameters
    window_size = 10
    walk_length = 40
    batch_size=128
    embedding_size = 128
    walk_per_vertex=80
    learning_rate=0.025
    momentum=0.9
    random_state = 89
    

    G = utils.load_matfile(file=data_dir + '/' + file_name,
                           variable_name='network')
    G = Graph(graph=G,
              window_size=window_size,
              walk_length=walk_length,
              random_state=random_state)

    G_loaded = DataLoader(G,
                          batch_size=batch_size,
                          drop_last=False, shuffle=True)

    model = deepwalk(G, embedding_size=embedding_size)
    model = model.to(model.device)

    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    records = defaultdict(list)
    start = datetime.now()

    for epoch in range(walk_per_vertex):
        losses = []
        model.train()
        for batched in G_loaded:
            optimizer.zero_grad()
            walk = utils.move_to(batched['walk'], model.device)
            target = batched['target'].to(model.device)
            context = batched['context'].to(model.device)
            loss = model(walk, target, context)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0 or epoch == walk_per_vertex - 1:
            if epoch == 0:
                eta = (datetime.now() - start).total_seconds()
            else:
                eta = (datetime.now() - tmp_timestamp).total_seconds()
            eta = divmod(eta, 60)
            eta = str(int(eta[0])) + 'm ' + str(np.round(eta[1], 4)) + 's'
            print("[Epoch: %s] Loss: %.4f (ETA: %s)" %
                    (str(epoch + 1).zfill(len(str(walk_per_vertex))),
                     np.mean(losses),
                     eta))
            tmp_timestamp = datetime.now()
        records['train'].append(np.mean(losses))
    print(f'Processing Time: {datetime.now() - start}')
    plt.plot(records['train'])








# walk_per_vertex (int, optional): walk per vertex (𝛾). Defaults to 80.
# walk_per_vertex=80

# learning_rate (float, optional): learning rate (𝛼) in optimization.
#             Defaults to 0.025.
# learning_rate=0.025





