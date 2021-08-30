import os
import tarfile
from pathlib import Path
import shutil
import re
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import pandas as pd

def tar_training_set(path: str):
    """Preprocessing for Combined Training Data Set

    Split the combined file in order to align with README

    Args:
        path (str): file path of 'combined_*' files.

    Returns:
        None: 'tarining_set.tar' file will be saved in the path.
    """
    training_set_dir = os.path.join(path, 'training_set')

    # Create Output Folder
    Path(training_set_dir).mkdir(parents=True, exist_ok=True)

    # List up the 'combined*' files
    files = [os.path.join(path, f)
             for f in os.listdir(path)
             if bool(re.search('combined', f))]
    files.sort()

    # Set up the inputs
    i = 0
    keys = []
    chunk = []

    # Read 'combined' files and Write 'mv_*'
    for f in tqdm(files):
        with open(f, 'r') as file:
            data = file.readlines()
        for l in data:
            if bool(re.search(':', l)):
                idx = str(int(re.sub(":\n", "", l)) - 1)
                idx = idx.rjust(7, '0')
                out_file = os.path.join(training_set_dir,
                                        'mv_' + idx + '.txt')
                if i != 0:
                    with open(out_file, 'w') as out:
                        out.writelines(chunk) 
                    chunk = []
            chunk.append(l)
            i += 1

    # Final Set
    idx = str(int(idx) + 1)
    idx = idx.rjust(7, '0')
    out_file = os.path.join(training_set_dir,
                            'mv_' + idx + '.txt')
    with open(out_file, 'w') as out:
        out.writelines(chunk)
    
    # Make tar
    with tarfile.open(os.path.join(path, 'training_set.tar'), 'w') as tar_out:
        tar_out.add(training_set_dir)
    shutil.rmtree(training_set_dir)
    return None

def import_training_set(path: str,
                        n_users: int=2649429,
                        n_movies: int=17770,
                        training_file: str='training_set.tar'):
    """Import training data as sparse matrix from Netflix Prize

    It uses 'scipy.sparse.lil_matrix' and 'scipy.sparse.csr_matrix'.

    Args:
        path (str): file path of 'training_set.tar' files.
        n_users (int, optional): total number of users
        in the triaing data. Defaults to 2649429.
        n_movies (int, optional): total number of movies
        in the triaing data. Defaults to 17770.
        training_file (str, optional): file name of training set.
        Defaults to 'training_set.tar'.

    Returns:
        scipy.sparse.csr.csr_matrix: Ratings Sparse Matrix
    """    
    R = lil_matrix((n_users, n_movies), dtype=np.uint8)
    file = os.path.join(path, training_file)
    input_tar = tarfile.open(file)
    for m in tqdm(input_tar.getmembers()):
        f = input_tar.extractfile(m)
        if f is not None:
            d = f.readlines()
            f.close()
            for l in d:
                l = l.decode('utf-8')
                if bool(re.search(':', l)):
                    MovieID = int(re.sub(':\n', '', l)) - 1
                    continue
                else:
                    l = l.split(',')
                    CustomerID = int(l[0]) - 1
                    Rating = int(l[1])
                R[CustomerID, MovieID] = Rating
    R = csr_matrix(R)
    return R

def import_movie_info(path: str,
                      movie_file: str='movie_titles.txt') -> pd.DataFrame:
    """Import Movie Title Data of Netflix Prize

    Args:
        path (str): [description]
        movie_file (str, optional): [description]. Defaults to 'movie_titles.txt'.

    Returns:
        pd.DataFrame: [description]
    """    
    file = os.path.join(path, movie_file)
    with open(file, 'r', encoding="ISO-8859-1") as f:
        d = f.readlines()
    d = [re.sub('\n', '', m) for m in d]
    years = [re.sub(',', '', re.findall(',[0-9]*', m)[0])
             for m in d]
    titles = [re.sub('[0-9]*,', '', m) for m in d]

    movie_titles = pd.DataFrame({'YearOfRelease': years,
                                'Title': titles})
    movie_titles.index.name = 'MovieID'

    return movie_titles

def import_test_set(path: str,
                    file_name: str):
    file = os.path.join(path, file_name)
    with open(file, 'r') as f:
        d = f.readlines()
    d = [re.sub('\n', '', m) for m in d]
    l_movies = []
    l_users = []
    for l in d:
        if bool(re.search(':', l)):
            MovieID = int(re.sub(':', '', l)) - 1
            continue
        else:
            l = l.split(',')
            CustomerID = int(l[0]) - 1  
        l_movies.append(MovieID)
        l_users.append(CustomerID)
    test_set = pd.DataFrame({'MovieID': l_movies,
                             'CustomerID': l_users})
    return test_set

#-----------------------------------------------------------------------
if __name__ == '__main__':
    root_dir = 'netflix_data'
    tar_training_set(root_dir)
