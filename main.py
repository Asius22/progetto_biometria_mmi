import scipy.io as scipy
import os
import mne
from autoreject import get_rejection_threshold

import matplotlib.pyplot as pl
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
#                     *    CODICE FUNZIONANTE PER CREAZIONE GRSFIIC(import shcipy.io, pandas e mne)    *
# should be 21
nPart = 1
# should be 3
nEsp = 1
# filtering const
low_cut_freq = 0.5
high_cut_freq = 45
tstep = 1.0
# ICA properties
random_state = 42
ica_n_components = .99 # ica must mantain 99% of data
ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
# .mat file directory name
DIRECTORY_NAME = "RAW_PARSED"
cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]

info = mne.create_info(cols, 256, ch_types='eeg')
ten_twenty_montage = mne.channels.make_standard_montage("standard_1020")

for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        # load the .mat file
        mat = scipy.loadmat(f'{DIRECTORY_NAME}/s{i}_s{j}.mat')
        # fetch recording data
        raw_input = mat['recording']
        print(mat['events'][1:1])
        # create the raw array with correct metadata
        raw = mne.io.RawArray(list(map(lambda x: x /1000000, raw_input.transpose())), info)
        raw.info["bads"] += ["COUNTER", "INTERPOLATED", "...UNUSED DATA..."]
        raw.set_montage(ten_twenty_montage,  on_missing="ignore")
        eeg_channels = mne.pick_types(raw.info, eeg=True)

        # raw.plot(title="Raw Array", start=15, duration=5, order=eeg_channels, n_channels=len(eeg_channels), scalings='auto')
        # start digital filtering and plot result
        raw_filtered = raw.filter(low_cut_freq, high_cut_freq, "all")
        raw_filtered.compute_psd(fmax=100).plot()
        raw_filtered.plot(title="filtered", duration=60, order=eeg_channels, n_channels=len(eeg_channels), scalings='auto')

        raw_events = mne.io.RawArray(mat['events'])\
            .filter(low_cut_freq, high_cut_freq, 'all')
        events_ica = mne.make_fixed_length_events(raw_events, duration=tstep)
        epochs_ica = mne.Epochs(raw_filtered, events_ica, tmin=0.0, tmax=tstep, baseline=None, preload=True)
        reject = get_rejection_threshold(epochs_ica)
        ica.fit(epochs_ica, reject=reject, tstep=tstep)
        ica.plot_components()
