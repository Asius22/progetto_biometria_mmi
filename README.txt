======================================
HUMAN IDENTIFICATION USING EEG SIGNALS
======================================


----------------------
STRUTTURA DEL PROGETTO
----------------------

Il progetto si compone di 4 file e di 3 cartelle

per quanto riguarda le cartelle, sono state create a runtime per contenere i risultati dei vari step dell'implementazione.

I file, invece, sono stati strutturati in modo da contenere uno step per ogni file, ad eccezion fatta del file main e del
file event_reader, che fanno parte entrambi della fase di preprocessing dei dati, ma abbiamo creato il file event_reader
poichè il dataset non conteneva una struttura omogenea per gli eventi, più nello specifico ogni istanza di evento conteneva
informazioni in posizioni distanti e, per poter costruire in maniera agile la struttura 3d richiesta da mne per l'analisi
degli eventi, abbiamo dovuto normalizzare la struttura del dataset.

-----------
main.py
-----------
contiene la fase di preprocessing dei dati, la suddivisione in epochs per poter utilizzare ICA agilmente e
la chiamata al metodo ica per ridurre il rumore e distinguere i dati delle varie aree del cervello (dove possibile).


----------------
events_reader.py
----------------
come anticipato questo file contiene la logica dietro la normalizzazione dela struttura events contenuta nel dataset.
Uno studio approfondito del dataset e della struttura del cubo multidimensionale events è stata necessaria per poter
giungere al risultato voluto.

---------------------
feature_extraction.py
---------------------
questo file contiene la fase di feature extraction; grazie all'utilizzo di mne questa fase ha richiesto poche righe di codice
ma ha permesso di ottenere risultati soddisfacenti dal modello.
In questo file si rivede l'uso della classe Epochs di mne, questa classe infatti salva i risultati di ICA al suo interno
e si porta quindi i vari valori e canali per tutto lo svolgimento del progetto.

--------
model.py
--------

questo file contiene la logica che sta dietro alla struttura ed alla valutazione del modello.
i risultati ottenuti da esso sono stati riportati nel paper del perogetto. Occorre specificare che questo file è stato
estratto da google colab poichè i componenti del gruppo non disponevano di un hardware appropriato per far girare
TensorFlow e keras

==================
dir numpy_features
==================

questa cartella contiene le features estratte da ogni soggetto, in formati npy in modo da poter essere letto da nupmy
senza perdita di dati dovuta al reshaping. La directory contiene un file s(i) con i che va da 1 a 21, ovvero una per ogni soggetto.
per ogni soggetto sono contenuti 3 file di feature, una per ogni esperimento

============
RAW_PARSED
============

questa cartella contiene i file .mat del dataset.

======
EPOCHS
======
In questa directory sono contenuti i vari file *-epo.fif salvati dopo l'esecuzione di ICA e contenenti appunto il suo output
I dati di questa cartella sono stati cruciali per l'esecuzione della feature extraction
