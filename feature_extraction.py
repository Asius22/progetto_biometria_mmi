import os

import mne
import librosa
import numpy as np

nSub = 21  # numero di soggetti da campionare
nEsp = 3  # numero di sedute da campionare per ogni soggetto
event_mapping = ['IMAGE', 'COGNITIVE', 'SSVEPC', 'SSVEP', 'REST', 'EYES']

roi = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
       "F4"]  # nomi dei sensori(potrebbero funzionare)

for i in range(1, nSub + 1):  # tutti i soggetti da 1 a nSub compreso
    os.mkdir(f"FEATURES/s_{i}")
    for j in range(1, nEsp + 1):  # tutte le sedute da 1 a nEsp compreso
        epochs = mne.read_epochs(f"EPOCHS/s{i}_s{j}-epo.fif", preload=True)  # carica le epoche

        epochs_data = epochs.get_data()  # restituisce un numpy array 3D che contiene n_epochs, n_channels, n_timepoints

        # applico la trasformata di Fourier per passare il dato nel dominio delle frequenze
        fft_data = np.fft.fft(epochs_data, axis=2)

        audio_signal = np.concatenate(fft_data, axis=1)  # concatena tutti i canali

        audio_signal = np.abs(audio_signal).astype(float)  # converti il segnale da complex a float

        mfcc = librosa.feature.mfcc(y=audio_signal)  # passa il segnale normalizzato per estrarre le mfcc
        # normalizzo il dato attraverso la deviazione principale e standard
        mfcc_mean = np.mean(mfcc, axis=1)  # calcolo gli indici sull'asse del tempo
        mfcc_normalized = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

        flat_array = mfcc_normalized.reshape(mfcc_normalized.shape[0], -1)  # (n_samples, num_frames * num_coefficients)
        np.savetxt(f"FEATURES/s_{i}/s{i}_s{j}.csv", flat_array, delimiter=",", )  # salva il file
