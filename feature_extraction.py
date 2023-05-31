import os
import matplotlib.pyplot as plt
import mne
import pandas

nSub = 21  # numero di soggetti da campionare
nEsp = 3  # numero di sedute da campionare per ogni soggetto
fmin = 8   # minimo per onde alpha
fmax = 30  #massismo per le onde beta
n_fft = 256
n_overlap = 128
n_jobs = 1
os.mkdir(f"./FEATURES3")
for i in range(1, nSub + 1):  # tutti i soggetti da 1 a nSub compreso
    os.mkdir(f"./FEATURES3/s{i}")

    for j in range(1, nEsp + 1):  # tutte le sedute da 1 a nEsp compreso
        epochs = mne.read_epochs(f"EPOCHS/s{i}_s{j}-epo.fif", preload=True)  # carica le epoche
        epochs.drop_channels(['COUNTER', 'INTERPOLATED', '...UNUSED DATA...'])  # droppa bad channels
        # restituisce un numpy array 3D che contiene n_epochs, n_channels, n_timepoints/n_samples
        epochs_data = epochs.get_data()

        psd_df = epochs.compute_psd(tmax=None, fmax=45, fmin=4).to_data_frame()

        psd_df.drop(['condition', 'epoch', 'freq'], axis=1, inplace=True)

        psd_df.to_csv(f"./FEATURES3/s{i}/s{i}_s{j}.csv")