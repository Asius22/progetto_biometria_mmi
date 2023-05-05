import scipy.io as scipy
from matplotlib import pyplot as pt
import pandas

cols = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6",
        "F4", "...UNUSED DATA..."]
for i in range(1, 21):
    for j in range(1, 3):
        mat = scipy.loadmat(f'RAW_PARSED/s{i}_s{j}.mat') #apri tutti i file
        frame = pandas.DataFrame(data=mat["recording"], columns=cols) #converti il dataframe
        pt.hist(frame) # tentativo di istogramma
        pt.show()
# frame.to_csv("dataframe.csv")
