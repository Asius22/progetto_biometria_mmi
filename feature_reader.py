import pandas as pd
import matplotlib.pyplot as plt
import sklearn
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

nPart = 21  # numero di partecipanti e di classi
nEsp = 3  # numero di file per ogni classe
classes = [i for i in range(1, nPart + 1)]
data = pd.DataFrame()  # dati per le istanze train

for i in range(1, nPart + 1):
    for j in range(1, nEsp + 1):
        file = pd.read_csv(f'./FEATURES3/s{i}/s{i}_s{j}.csv')
        file['labels'] = i - 1
        data = pd.concat([data, file])

labels = data['labels']
data = data.drop('labels', axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.2)

# dati per il modello+
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_train.shape)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""

precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

recall = recall_score(y_test, y_pred,average='macro')
print("Recall:", recall)
"""
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
