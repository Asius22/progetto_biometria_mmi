import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import export_graphviz

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

nPart = 21  # numero di partecipanti e di classi
nEsp = 3  # numero di file per ogni classe
classes = [i for i in range(1, nPart + 1)]  # classi da associare aai soggetti
data = pd.DataFrame()  # dati per le istanze train
X_test = pd.DataFrame()  # dati per il testing
for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        file = pd.read_csv(f'./FEATURES3/s{i}/s{i}_s{j}.csv').T  # leggi la trasposta per avere ogni channel sulla riga
        file['labels'] = i - 1  # aggiunge la classe per ogni riga da campionare
        data = pd.concat([data, file])  # concatena per avere un dataframe con tutti gli esperimenti

data = data.drop('Unnamed: 0')  # colonna inutile
labels = data['labels']  # classi associate ad ogni istanza
data = data.drop('labels', axis=1)  # dati "puri"

# split 80-20
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.2)

scaler = sklearn.preprocessing.MinMaxScaler()
# normalizza i dati usando la strategia MinMax
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()  # random forest
fitted = rf.fit(X_train, y_train)  # addestra il classificatore
y_pred = rf.predict(X_test)  # prova una previsione
report = classification_report(y_test, y_pred)
print(report)
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()  # mostra la confusion matrix
