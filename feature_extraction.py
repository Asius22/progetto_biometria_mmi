import os
import mne
import numpy as np

nSub = 21  # numero di soggetti da campionare (21)
nEsp = 3  # numero di sedute da campionare per ogni soggetto (3)
fmin = 8  # minimo per onde alpha
fmax = 80  # massismo per le onde beta
directory = "./NUMPY_FEATURES"  # cartella in cui salvare i file
os.mkdir(directory)  # crea la cartella
for i in range(1, nSub + 1):  # tutti i soggetti da 1 a nSub compreso
    os.mkdir(f"{directory}/s{i}")
    for j in range(1, nEsp + 1):  # tutte le sedute da 1 a nEsp compreso
        epochs = mne.read_epochs(f"EPOCHS/s{i}_s{j}-epo.fif", preload=True)  # carica le epoche
        epochs.drop_channels(['COUNTER', 'INTERPOLATED', '...UNUSED DATA...'])  # droppa bad channels
        # restituisce un numpy array 3D che contiene n_epochs, n_channels, n_timepoints/n_samples
        epochs_data = epochs.get_data()  # leggi i dati delle epoche

        psd_df = epochs.compute_psd(method='welch', tmax=None, fmax=80, fmin=4)  # calcola psd usando il metodod welch

        psd_npy = np.array(psd_df)
        np.save(f"{directory}/s{i}/s{i}_s{j}.npy", psd_npy)  # salva il file per il prossimo step (RNN)