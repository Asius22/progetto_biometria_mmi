import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers
from keras.models import Sequential
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
#grid search


nPart = 21  # numero di partecipanti e di classi
nEsp = 3  # numero di file per ogni classe
classes = [i for i in range(0, nPart )]
#data = pd.DataFrame()  # dati per le istanze train
labels = np.array([])
data = np.array([])
for i in range(1, nPart+1):
  for j in  range(1, nEsp + 1):

    file = np.load(f'/content/drive/MyDrive/Colab Notebooks/NUMPY_FEATURES/s{i}/s{i}_s{j}.npy')
    if i==1 and j == 1:
      data = file # il primo file non va concatenato
    else:
      data = np.concatenate((data, file), axis=0) # tutti gli altri concatenali sull'asse 0 (verticale)
    labels = np.concatenate((labels, np.full((file.shape[0], ), i-1)), axis= 0)

scaler = sklearn.preprocessing.PowerTransformer()
# normalizzazione dei dati

for i in range (data.shape[1]):
  data[:, i, :] = scaler.fit_transform(data[:, i, :]) # normalizza i valori per ogni epoca al timestep i
print("dati normalizzati... ")

# divisione dei dati
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.4)
print("dati divisi... ")


def create_model(lstm_dropout = 0.1, dropout_rate=0.3, dense_unit=64, lstm_unit=32):
    model = Sequential([
      layers.LSTM(lstm_unit,
                  activation='relu',
                  input_shape=(X_train.shape[1], X_train.shape[2]),
                  return_sequences = False,
                  dropout=lstm_dropout, # dropout tra input e stati nascosti
                  recurrent_dropout=lstm_dropout),# dropout tra gli strati nascosti
      layers.BatchNormalization(beta_regularizer='l1'),
      layers.Dense(dense_unit, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(21, activation='softmax')
  ])
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# dati per il fit
batch_size=63 #grandezza del batch
epochs=70 # numero di epoche da utilizzare

media = 0.0 # utilizzata per avere un'accuracy media da poter comunicare
nTest = 10 # numero di test per calcolare l'accuracy media

for i in range(0, nTest): # ad ogni iterazione crea un modello, fittalo e valutane l'accuracy
  # creazione modello
  model = create_model()

  callback = EarlyStopping(
      monitor='val_loss', #monitora la val_loss
      min_delta=0.01, #anche il minimo miglioramento va bene
      patience=7, #se non trova un miglioramento ini 10 epoche stoppa la run
      mode='min', #i miglioramenti si calcolano sul minimo valore
      restore_best_weights=True
  )

  model.fit(X_train,
              y_train,
              validation_data=(X_test, y_test),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[callback]

  )
  accuracy = model.evaluate(X_test, y_test) # valuta il modello
  print(f"fine esecuzione: {i}")
  y_pred = model.predict(X_test) # fai un predict per poter utilizzare classifixation_report
  print(classification_report(y_test, np.argmax(y_pred, axis=1), labels=classes))
  media += accuracy[1]
media /= nTest
print(f"accuracy: {media}")


