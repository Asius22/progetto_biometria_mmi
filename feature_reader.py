import numpy as np
import pandas

for i in range(1, 22):
    for j in range(1, 4):
        file = pandas.read_csv(f"FEATURES/s{i}_s{j}.csv", sep=",")
        print(file.shape)
