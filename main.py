import scipy.io as scipy
import mne
from autoreject import get_rejection_threshold
from events_reader import read_event

# should be 21
nPart = 21
# should be 3
nEsp = 3
# filtering const
low_cut_freq = 1
high_cut_freq = 45
tstep = 5.0
# ICA properties
random_state = 42
ica_n_components = 14 # ica must mantain 99% of data
# .mat file directory name
DIRECTORY_NAME = "RAW_PARSED"
cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]

info = mne.create_info(cols, 256, ch_types='eeg')
ten_twenty_montage = mne.channels.make_standard_montage("standard_1020")

for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
        path = f'{DIRECTORY_NAME}/s{i}_s{j}.mat'
        # load the .mat file
        mat = scipy.loadmat(path)
        # fetch recording data
        raw_input = mat['recording']
        # carica eventi
        events = mat['events']
        event_mapping, event_array = read_event(path)

        # create the raw array with correct metadata
        raw = mne.io.RawArray(list(map(lambda x: x /1000000, raw_input.transpose())), info)
        raw.info["bads"] += ["COUNTER", "INTERPOLATED", "...UNUSED DATA..."]
        raw.set_montage(ten_twenty_montage,  on_missing="ignore")
        eeg_channels = mne.pick_types(raw.info, eeg=True)
        # plot pre filtering
        # raw.plot(title="Raw Array", start=15, duration=5, order=eeg_channels, n_channels=len(eeg_channels), scalings='auto')
        # start digital filtering and plot result
        raw_filtered = raw.copy().filter(low_cut_freq, high_cut_freq, "all")
        # raw_filtered.compute_psd(fmax=100).plot()
        # plot post filtering
        # raw_filtered.plot(title="filtered", duration=5, start=15, order=eeg_channels, n_channels=len(eeg_channels), scalings='auto')
        # create epochs with recording and events properly preprocessed
        epochs_ica = mne.Epochs(raw_filtered, event_array, event_id=event_mapping, tmin=0, tmax=tstep, baseline=None, preload=True, event_repeated='merge')
        reject = get_rejection_threshold(epochs_ica)
        # fit data to epochs
        ica.fit(epochs_ica, reject=reject, tstep=tstep)
        # remove selected artifact
        epochs_post_ica = ica.apply(epochs_ica.copy())
        # epochs_post_ica.plot()
        # ica.plot_components()
        # ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': high_cut_freq});
        # ica.plot_overlay(raw_filtered, picks="eeg")
        # ica.plot_components()

        # save epochs
        epochs_post_ica.save(f"./EPOCHS/s{i}_s{j}-epo.fif", overwrite=True)
        # save ica dates
        # ica.save(f"./ICA/s{i}_s{j}-ica.fif", overwrite=True)
