import numpy as np
import matplotlib.pyplot as pl
import shutil as os

for i in range(1, 22):
    print(np.load(f"FEATURES_2/s{i}/s{i}_s1.npy").shape)

