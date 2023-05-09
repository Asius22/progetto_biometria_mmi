from pyemd import emd
import scipy.io as scipy
import pandas
import mne
nPart = 1
nEsp = 3
DIRECTORY_NAME = "RAW_PARSED"
cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]
info = mne.create_info(cols, 250)
mat = scipy.loadmat(f'{DIRECTORY_NAME}/s1_s1.mat') #apri tutti i file
frame = pandas.DataFrame(data=mat["recording"], columns=cols).transpose() #converti il dataframe e calcola la trasposta per avere i vari canali sulle righe
emd_data = emd(frame)

info = mne.create_info(cols, 250)
raw = mne.io.RawArray(emd_data, info)

raw.info["bads"] += ["COUNTER", "INTERPOLATED","...UNUSED DATA..." ]

raw.plot()
"""
for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        mat = scipy.loadmat(f'{DIRECTORY_NAME}/s{i}_s{j}.mat') #apri tutti i file
        frame = pandas.DataFrame(data=mat["recording"], columns=cols) #converti il dataframe
        info = mne.create_info(cols, 250)
        emd_frame = emd.imftools.amplitude_normalise( pandas.DataFrame.to_numpy(frame.transpose()))
        raw = mne.io.RawArray(emd_frame, info)
        raw.info["bads"] += ["COUNTER", "INTERPOLATED","...UNUSED DATA..." ]

        raw.plot()

"""