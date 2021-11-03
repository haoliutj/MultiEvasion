import copy, lief, math,operator,struct
import numpy as np
from attacks import attack_utils

data_path = '../data/test.exe'
first_n_byte=102400
preferable_extension_amount = 512

with open(data_path,'rb') as f:
    # tmp = [i for i in f.read()[:first_n_byte]]
    tmp = [i for i in f.read()]

X = copy.deepcopy(tmp)

f_byte = bytearray(list(np.array(X,dtype=int)))

X_real = b''.join([bytes([i]) for i in X])

with open('../data/test_sample.exe','wb') as f:
    f.write(X_real)

print(f'file wrote!')