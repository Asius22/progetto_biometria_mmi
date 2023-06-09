import scipy.io as scipy
import numpy as np

"""
    # ciclo per creare l'event:dict
    # deve contenere la label dell'evento e l'id dell'evento
    # il tipo dell'evento si trova nel percorso 2 0 0 0 0
"""
event_mapping = {'IMAGE': 1, 'COGNITIVE': 5, 'SSVEPC': 201, 'SSVEP': 211, 'REST': 3, 'EYES': 111}

"""
    funzione creata per mettere le informazioni utili tutte nella stessa posizione
"""
def normalizeParsed():
    n_pers = 21
    n_esp = 3
    for i in range(1, n_pers + 1):  # apri tutti i file
        for j in range(1, n_esp + 1):
            save_mat = False
            mat = scipy.loadmat(f"RAW_PARSED/s{i}_s{j}.mat")
            print(f"-------------------s{i}_s{j}--------------------------------")
            events = mat['events']  # carica gli eventi
            for event in events:  # gli eventi sono o in posizione 2 0 0 0 0 o in posizione 2 0 0 3 0
                final_event_type = event[2][0][0][0][0]
                if final_event_type.dtype.type is not np.str_ or not event_mapping.__contains__(
                        final_event_type):  # controlla che in tutti i file l'occorrenza 2 0 0 0 0 sia una stringa
                    save_mat = True
                    print(event[2][0][0])
                    try:
                        if type(event[2][0][0][1][0]) is np.str_ and event_mapping.__contains__(event[2][0][0][1][0]):
                            event[2][0][0][0], event[2][0][0][1] = event[2][0][0][1], event[2][0][0][0]
                            # print("è 1")
                        elif type(event[2][0][0][2][0]) is np.str_ and event_mapping.__contains__(event[2][0][0][2][0]):
                            event[2][0][0][0], event[2][0][0][2] = event[2][0][0][2], event[2][0][0][0]
                            # print("è 2")
                        else:
                            event[2][0][0][0], event[2][0][0][3] = event[2][0][0][3], event[2][0][0][0]
                            # print("è 3")
                    except IndexError:
                        print(f"----------------{event}")

            if save_mat:
                scipy.savemat(f"RAW_PARSED/s{i}_s{j}.mat", mdict=mat)


def read_event(path):
    event_array = []

    print(f"-----------{path}----------")
    events = scipy.loadmat(path)['events']
    event_start = events[0][0][0][0]  # tempo di inizio dell'esperimento
    for event in events:
        # il tipo di evento, post normalizzazione, si trova in posizione 2-> 0-> 0-> 0-> 0
        event_type = event[2][0][0][0][0]

        first_column = event[0][0][0] - event_start  # inizio evento
        second_column = event[1][0][0] - event_start  # fine evento
        # la terza colonna contiene il tipo di evento che si è presentato
        third_column = event_mapping[event_type] # tipo evento (image, cognitive ecc... 9
        array = np.array([first_column, second_column, third_column], dtype=np.int64)
        event_array.append(array)

    event_array.sort(key=lambda x: x[0])  # ordina gli eventi in base al tempo di inizio
    return event_mapping, np.asarray(event_array)
