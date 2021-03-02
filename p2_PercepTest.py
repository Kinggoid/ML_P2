import unittest

from ML.p2_perceptronen.p2_perceptronen import Perceptron


class MyTestCase(unittest.TestCase):
    def test_And(self):
        """In deze functie gaan we eerst een Perceptron met random parameters maken. Vervolgens gaan we deze perceptron
        trainen tot het voor elk van de verschillende inputs de juiste waardes geeft. Als dit lukt, werkt de AND
        perceptron."""

        AND = Perceptron([0, 4], 1, 'AND gate', 0.2)
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        verwachte_outputs = [0, 0, 0, 1]

        tries = 100  # Het kan zijn dat het onmogelijk is voor een perceptron om de juiste parameters te krijgen voor
        # elke output. Aangezien we onze initiële parameters niet zo enorm hebben gemaakt, moet een
        # mogelijke perceptron makkelijk de juiste parameters hebben gekregen binnen deze 100 kansen.

        good = None  # Hoeveel inputs geven het juiste antwoord
        while good != 4:  # Pas wanneer alle inputs een juist antwoord geven, gaan we uit de while loop
            good = 0
            for i in range(0, len(inputs)):  # Hieronder berekenen we de uitkomst van de perceptron. Als hier de juiste
                                             # uikomst uit komt zeggen we dat deze goed is. Maar het niet juist is,
                                             # updaten we de perceptron.
                uitkomst = AND.calculate_output([inputs[i][0], inputs[i][1]])
                if uitkomst == verwachte_outputs[i]:
                    good += 1
                else:
                    AND.update([inputs[i][0], inputs[i][1]], verwachte_outputs[i])

            tries -= 1
            if tries == 0:  # Als er geen kansen meer zijn is het een onmogelijk voor deze perceptron om juiste
                            # parameters te vinden om de gegeven uitkomsten te krijgen.
                break

        self.assertEqual(good, 4)  # Kijk of de outputs goed zijn


    def test_XOR(self):
        """In deze functie gaan we eerst een Perceptron met random parameters maken. Vervolgens gaan we deze perceptron
        trainen tot het voor elk van de verschillende inputs de juiste waardes geeft. Als dit lukt, werkt de XOR
        perceptron."""

        XOR = Perceptron([0, 4], 1, 'XOR gate', 0.2)
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        verwachte_outputs = [0, 1, 1, 0]

        tries = 100  # Het kan zijn dat het onmogelijk is voor een perceptron om de juiste parameters te krijgen voor
                     # elke output. Aangezien we onze initiële parameters niet zo enorm hebben gemaakt, moet een
                     # mogelijke perceptron makkelijk de juiste parameters hebben gekregen binnen deze 100 kansen.

        good = None  # Hoeveel inputs geven het juiste antwoord
        while good != 4:  # Pas wanneer alle inputs een juist antwoord geven, gaan we uit de while loop
            good = 0

            for i in range(0, len(inputs)):  # Hieronder berekenen we de uitkomst van de perceptron. Als hier de juiste
                                             # uikomst uit komt zeggen we dat deze goed is. Maar het niet juist is,
                                             # updaten we de perceptron.
                uitkomst = XOR.calculate_output([inputs[i][0], inputs[i][1]])
                if uitkomst == verwachte_outputs[i]:
                    good += 1
                else:
                    XOR.update([inputs[i][0], inputs[i][1]], verwachte_outputs[i])

            tries -= 1
            if tries == 0:  # Als er geen kansen meer zijn is het een onmogelijk voor deze perceptron om juiste
                            # parameters te vinden om de gegeven uitkomsten te krijgen.
                break

        self.assertEqual(good, 4)  # Kijk of de outputs goed zijn

        # Deze test zal altijd falen. Dit komst niet doordat er een fout staat in de code maar omdat het onmogelijk
        # is voor één perceptron om een XOR gate na te bootsen. Dit komt omdat een perceptron alleen lineaire problemen
        # op kan lossen en dit niet een lineair probleem is.


if __name__ == '__main__':
    unittest.main()
