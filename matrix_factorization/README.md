# Recomender Systems Summary and Implementation

The 'README.md'file in each folder is a short summary of each paper.
For the detailed summary, please refer to the external link in the file.
The implementation code is in each folder as well.

The code can be run as following process.

```python
python main.py
```

```python
# main.py

if __name__ == '__main__':
    '''
    # Import Movie Lens Data
    ----------------------------------------------------------------
    '''
    movielens_dir = './movielens_data'

    # Training set Preparation
    with open(os.path.join(movielens_dir, 'u1.base'), 'r') as f:
        data = f.readlines()

    train_u = []
    train_i = []
    train_r = []
    for l in data:
        l = re.sub('\n', '', l)
        l = l.split('\t')
        train_u.append(np.int64(l[0])-1)
        train_i.append(np.int64(l[1])-1)
        train_r.append(np.int64(l[2]))

    train = sparse.csr_matrix((train_r, (train_u, train_i)))

    # Test set preparation
    with open(os.path.join(movielens_dir, 'u1.test'), 'r') as f:
            data = f.readlines()

    test_u = []
    test_i = []
    test_r = []
    for l in data:
        l = re.sub('\n', '', l)
        l = l.split('\t')
        test_u.append(np.int64(l[0])-1)
        test_i.append(np.int64(l[1])-1)
        test_r.append(np.int64(l[2]))

    test = sparse.csr_matrix((test_r, (test_u, test_i)))

    '''
    # Run
    ----------------------------------------------------------------
    '''
    # Algorithm should be changed!
    print('Algorithm Name --------------------------')
    model = algorithm.algorithm(train, test, random_state=89)
    model.fit()
```

