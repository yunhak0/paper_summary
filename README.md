# Papar Summary of Recommender Systems & Graph Neural Networks

Paper Summary and Self-written code of Recommender Systems

## Recommender Systems

#### Explicit Feedback

* Matrix Factorization based
    + [Factorization Meets the Neighborhood](https://dl.acm.org/doi/10.1145/1401890.1401944) (Y Koren, KDD'08) [[SVD++](rec_sys/matrix_factorization/SVD_integrated)]
    + [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf) (A Mnih, RR Salakhutdinov, NIPS'07) [[PMF](rec_sys/matrix_factorization/SVD_integrated/PMF)]

#### Implicit Feedback

* Matrix Factorization based
    + [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) (Y Hu et al., ICDM'08) ~~[[OCCF]()]~~ (TBU)
    + [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) (S Rendle et al., arXiv'12) ~~[[BPR]()]~~ (TBU)
* Metric Learning based
    + [Collaborative Metric Learning](https://dl.acm.org/doi/10.1145/3038912.3052639) (CK Hsieh et al., WWW'17) ~~[[CML]()]~~ (TBU)
* Neural Network based
    + [Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569) (X He et al., WWW'17) ~~[[NCF]()]~~ (TBU)

#### Side Information (Social Network)

* [SoRec: social recommendation using probabilistic matrix factorization](https://dl.acm.org/doi/10.1145/1458082.1458205) (H Ma et al., CIKM'08) ~~[[SoRec]()]~~ (TBU)
* [Recommender Systems with Social Regularization](https://dl.acm.org/doi/10.1145/1935826.1935877) (H Ma et al., WSDM'11) ~~[[SoReg]()]~~ (TBU)

#### Deep Learning based Recommender Systems

* [CDL: Collaborative Deep Learning for Recommender Systems](http://www.wanghao.in/paper/KDD15_CDL.pdf) (H Wang et al., KDD'15) ~~[[CDL]()]~~ (TBU)
* [AutoRec: Autoencoders Meets Collborative Filtering](https://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf) (S Sedhain et al., WWW'15) ~~[[AutoRec]()]~~ (TBU)
* [YouTube: Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf) (P Covingtion et al., RecSys'16) ~~[[YouTube]()]~~ (TBU)
* [Wide&Deep: Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) (H-T Cheng et al. DLRS'16) ~~[[Wide&Deep]()]~~ (TBU)
    + [FM: Factorization Machines](https://ieeexplore.ieee.org/document/5694074) (S Rendle, ICDM'10) ~~[[FM]()]~~ (TBU)


## Graph Neural Networks

#### Random Walk

* [Deepwalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) (B Perozzi et al., KDD'14) [[Deepwalk](gnn/deepwalk)]
* [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) (A Grover, J Leskovec, KDD'16) [[node2vec](gnn/node2vec)]
* [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf) (J Tang et al., WWW'15) ~~[[LINE]()]~~ (TBU)
* [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) (Y Dong et al., KDD'17) ~~[[metapath2vec]()]~~ (TBU)

#### Graph Neural Network

* Supervised
    + [GCN: Semi-Supervised Classification with Graph Convolution Networks](https://openreview.net/pdf?id=SJU4ayYgl) (TN. Kipf, M Welling, ICLR'17) ~~[[GCN]()]~~ (TBU)
    + [GAT: Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) (P Velickovic et al., ICLR'18) ~~[[GAT]()]~~ (TBU)

* Unsupervised
    + [GVAE: Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) (TN. Kipf, M Welling, arXiv) ~~[[GVAE]()]~~ (TBU)
        + [VAE: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (D P Kingma, M Welling, arXiv) ~~[[VAE]()]~~ (TBU)
    + [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf) (W L. Hamilton et al., NIPS'17) ~~[[GraphSAGE]()]~~ (TBU)

#### Knowledge Graph Embedding

* [TransE: Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) (A Bordes et al, NIPS'13) ~~[[TransE]()]~~ (TBU)
