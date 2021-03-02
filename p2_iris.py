import numpy as np


def irissen(planten, data, perceptron):
    """In deze functie krijgen we een dataset binnen met verschillende parameters van meerdere irissen. Hier gaan
    we proberen de gegeven perceptron het onderscheid tussen de irissen te laten leren."""
    targets = [np.where(data.target_names == i.lower()) for i in planten]  # Welke plantensoorten gaat de perceptron onderscheiden.

    lengte = 50 * len(targets)  # Elke soort heeft 50 planten in de dataset, dus hier kijken we hoeveel planten we steeds langs gaan.

    good = None  # Hoeveel inputs geven het juiste antwoord
    while good != lengte:  # Pas wanneer alle inputs een juist antwoord geven, gaan we uit de while loop
        good = 0
        for i in range(0, lengte):  # Hieronder berekenen we de uitkomst van de perceptron. Als hier de juiste uikomst uit
                                 # komt zeggen we dat deze goed is. Maar het niet juist is, updaten we de perceptron.
            target = data.target[i]
            if target in targets:
                ding = data.data[i]
                uitkomst = perceptron.calculate_output([ding[0], ding[1], ding[2], ding[3]])
                if uitkomst == target:
                    good += 1
                else:
                    perceptron.update([ding[0], ding[1], ding[2], ding[3]], target)
