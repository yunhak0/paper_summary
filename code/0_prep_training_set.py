import os
import re
from tqdm import tqdm

files = [os.getcwd() + '/data/' + f
         for f in os.listdir('data')
         if bool(re.search('combined', f))]
files.sort()

# Split the combined file in order to align with README
i = 0
keys = []
chunk = []
for f in tqdm(files):
    with open(f, 'r') as file:
        data = file.readlines()
    for l in data:
        if bool(re.search(':', l)):
            idx = str(int(re.sub(":\n", "", l)) - 1)
            out_file = os.path.join(os.getcwd(), 'data',
                                    'training_set_' + idx + '.txt')
            if i != 0:
                with open(out_file, 'w') as out:
                    out.writelines(chunk) 
                chunk = []
        chunk.append(l)
        i += 1

# Final Set
idx = str(int(idx) + 1)
out_file = os.path.join(os.getcwd(), 'data',
                        'training_set_' + idx + '.txt')
with open(out_file, 'w') as out:
    out.writelines(chunk)
