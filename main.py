import scipy.io as scipy
import mne
from autoreject import get_rejection_threshold
from events_reader import read_event

nPart = 21  # numero di partecipanti (21)
nEsp = 3  # numero di esperimenti per partecipante (3)
# filtering const
low_cut_freq = 1  # frequenza minima da campionare
high_cut_freq = 100  # frequenza massima da campionare
tstep = 5.0  # grandezza delle epoche di campionamento (in secondi)
# ICA properties
random_state = 42
ica_n_components = .99  # ica must mantain 99% of data
# .mat file directory name
DIRECTORY_NAME = "RAW_PARSED"
cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]  # nome delle colonne del .mat

info = mne.create_info(cols, 256, ch_types='eeg')   # metadati dell'esperimento
ten_twenty_montage = mne.channels.make_standard_montage("standard_1020")  # standard utilizzato nell'esperimento

for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        # classe ICA per il filtering
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
        #carica il file .mat
        path = f'{DIRECTORY_NAME}/s{i}_s{j}.mat'
        mat = scipy.loadmat(path)
        raw_input = mat['recording']  # misurazioni dell'esperimento
        events = mat['events']  # dati degli eventi
        event_mapping, event_array = read_event(path)  # converti gli eventi nel formato voluto da mpi

        # crea il raw array con i giusti metadati
        raw = mne.io.RawArray(list(map(lambda x: x / 1000000, raw_input.transpose())), info)  # converti il segnale in micro volt
        raw.info["bads"] += ["COUNTER", "INTERPOLATED", "...UNUSED DATA..."]  # colonne che non ci servono
        raw.set_montage(ten_twenty_montage, on_missing="ignore")  # metadata
        eeg_channels = mne.pick_types(raw.info, eeg=True)  # metadata
        # plot pre -iltering di 5 secondi
        raw.plot(title="Raw Array", start=15, duration=5, order=eeg_channels, n_channels=len(eeg_channels), scalings='auto')

        raw_filtered = raw.copy().filter(low_cut_freq, high_cut_freq, "all")  # primo filtro low-max
        # plot post-filtering di 5 secondi dell'esperimento
        raw_filtered.plot(title="filtered", duration=5, start=15, order=eeg_channels, n_channels=len(eeg_channels), scalings='auto')
        # crea epoche con raw ed eventi preprocessati
        epochs_ica = mne.Epochs(raw_filtered, event_array, event_id=event_mapping, tmin=0, tmax=tstep, baseline=None,
                                preload=True, event_repeated='merge')
        reject = get_rejection_threshold(epochs_ica)
        ica.fit(epochs_ica, reject=reject, tstep=tstep)  # fitta i dati in modo da creare i canali indipendenti ICA

        # remove selected artifact
        epochs_post_ica = ica.apply(epochs_ica.copy())  # rimuovi il rumore
        # plot vari
        epochs_post_ica.plot()
        ica.plot_components()
        ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': high_cut_freq});
        ica.plot_overlay(raw_filtered, picks="eeg")
        ica.plot_components()

        # salva epoche per il prossimo step (feature extraction)
        epochs_post_ica.save(f"./EPOCHS/s{i}_s{j}-epo.fif", overwrite=True)
