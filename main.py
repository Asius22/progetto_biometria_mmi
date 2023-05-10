import scipy.io as scipy
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne.preprocessing import(
        create_ecg_epochs,
        create_eog_epochs,
        compute_proj_ecg,
        compute_proj_eog,
)

#                     *    CODICE FUNZIONANTE PER CREAZIONE GRSFIIC(import shcipy.io, pandas e mne)    *

nPart = 21
nEsp = 3
DIRECTORY_NAME = "RAW_PARSED"
cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]
info = mne.create_info(cols, 250)
mat = scipy.loadmat(f'{DIRECTORY_NAME}/s1_s1.mat') #apri tutti i file
frame = pandas.DataFrame(data=mat["recording"], columns=cols) #converti il dataframe e calcola la trasposta per avere i vari canali sulle righe


for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        mat = scipy.loadmat(f'{DIRECTORY_NAME}/s{i}_s{j}.mat') #apri tutti i file
        frame = pandas.DataFrame(data=mat["recording"], columns=cols) #converti il dataframe
        info = mne.create_info(cols, 250)
        raw = mne.io.RawArray(frame.transpose(), info)
        raw.info["bads"] += ["COUNTER", "INTERPOLATED","...UNUSED DATA..."]
        raw.crop(tmax=60, tmin=1)
        raw.plot()
"""
        raw_check = raw.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
            return_scores=True, verbose=True)
        print(auto_noisy_chs)  # we should find them!
        print(auto_flat_chs)  # none for this dataset
"""

