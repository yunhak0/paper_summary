import os
import tarfile
from pathlib import Path
import shutil
import re
from tqdm import tqdm

# Split the combined file in order to align with README
def tar_training_set(root_dir):
    training_set_dir = os.path.join(root_dir, 'training_set')

    # Create Output Folder
    Path(training_set_dir).mkdir(parents=True, exist_ok=True)

    # List up the 'combined*' files
    files = [os.path.join(root_dir, f)
             for f in os.listdir(root_dir)
             if bool(re.search('combined', f))]
    files.sort()

    # Set up the inputs
    i = 0
    keys = []
    chunk = []

    # Read 'combined' files and Write 'training_set_*'
    for f in tqdm(files):
        with open(f, 'r') as file:
            data = file.readlines()
        for l in data:
            if bool(re.search(':', l)):
                idx = str(int(re.sub(":\n", "", l)) - 1)
                out_file = os.path.join(training_set_dir,
                                        'training_set_' + idx + '.txt')
                if i != 0:
                    with open(out_file, 'w') as out:
                        out.writelines(chunk) 
                    chunk = []
            chunk.append(l)
            i += 1

    # Final Set
    idx = str(int(idx) + 1)
    out_file = os.path.join(training_set_dir,
                            'training_set_' + idx + '.txt')
    with open(out_file, 'w') as out:
        out.writelines(chunk)
    
    # Make tar
    with tarfile.open(os.path.join(root_dir, 'training_set.tar'), 'w') as tar_out:
        tar_out.add(training_set_dir)
    shutil.rmtree(training_set_dir)
    return None

#-----------------------------------------------------------------------
if __name__ == '__main__':
    root_dir = 'data'
    tar_training_set(root_dir)
