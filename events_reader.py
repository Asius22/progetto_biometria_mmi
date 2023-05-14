import scipy.io as scipy

image_duration=5
rest_duration=120


events = scipy.loadmat("RAW_PARSED/s1_s1.mat")['events']
event_dict = []
event_array = []
"""
    # ciclo per creare l'event:dict
    # deve contenere la label dell'evento e l'id dell'evento
    # il tipo dell'evento si trova nel percorso 2 0 0 0 0
"""
#types = ['IMAGE', 'COGNITIVE', 'SSVEPC', 'SSVEP', 'REST', 'EYES']
types = dict()

for i in range(0, len(events)):
    event = events[i]
    type = event[2][0][0][0][0]

    print(event[2])
    key = f"{type} {i}"
    event_dict.append({key: i})

