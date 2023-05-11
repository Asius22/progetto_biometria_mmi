import scipy.io as scipy
import pandas
import os
import matplotlib.pyplot as pl
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
#                     *    CODICE FUNZIONANTE PER CREAZIONE GRSFIIC(import shcipy.io, pandas e mne)    *
# should be 21
nPart = 1
# should be 3
nEsp = 3
DIRECTORY_NAME = "RAW_PARSED"
cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]

effectiveCols = cols[2:16]
fine_cal_file = os.path.join('electrode-locations.dat')
transformer = FastICA(n_components=None, whiten='unit-variance', random_state=0)

for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        mat = scipy.loadmat(f'{DIRECTORY_NAME}/s{i}_s{j}.mat') #apri tutti i file
        frame = pandas.DataFrame(data=mat["recording"], columns=cols) #converti il dataframe
        frame = frame.drop(columns=["COUNTER", "INTERPOLATED", "...UNUSED DATA..."])
        pl.plot(frame)
        pl.show()
        frame_transformed = transformer.fit_transform(frame)
        pl.plot(frame_transformed)
        pl.show()
