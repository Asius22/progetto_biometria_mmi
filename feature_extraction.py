import os
import matplotlib.pyplot as plt
import mne
import librosa
from librosa import feature
import numpy as np

nSub = 1  # numero di soggetti da campionare
nEsp = 1  # numero di sedute da campionare per ogni soggetto

for i in range(1, nSub + 1):  # tutti i soggetti da 1 a nSub compreso
    # os.mkdir(f"FEATURES_2/s{i}")
    for j in range(1, nEsp + 1):  # tutte le sedute da 1 a nEsp compreso
        epochs = mne.read_epochs(f"EPOCHS/s{i}_s{j}-epo.fif", preload=True)  # carica le epoche
        epochs.drop_channels(['COUNTER', 'INTERPOLATED', '...UNUSED DATA...'])  # droppa bad channels
        # restituisce un numpy array 3D che contiene n_epochs, n_channels, n_timepoints/n_samples
        epochs_data = epochs.get_data()

        fft = np.fft.fft(epochs_data)  # applico la trasformata di fourier
        fft = np.concatenate(fft, axis=1)  # concatena i dati lasciando inalterati i canali
        audio_signal = fft.astype(float)  # parse float
        mfcc = librosa.feature.mfcc(y=audio_signal, sr=epochs.info['sfreq'])  # calcola mfcc
        mfcc_scaled = np.mean(mfcc.T, axis=0)  # calcola la media sulla trasposta

        # salva il file
        np.save(f"FEATURES_2/s{i}/s{i}_s{j}.npy", mfcc_scaled) # salva la shape (17.20.166)

