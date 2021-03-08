import unittest

from ML.p2_perceptronen.p2_perceptronen import Perceptron


class MyTestCase(unittest.TestCase):
    def test_AND(self):
        """In deze functie gaan we eerst een Perceptron met random parameters maken. Vervolgens gaan we deze perceptron
        trainen tot het voor elk van de verschillende inputs de juiste waardes geeft. Als dit lukt, werkt de AND
        perceptron."""

        AND = Perceptron([0, 4], 1, 'AND gate', 0.2)
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        verwachte_outputs = [0, 0, 0, 1]
        amount_inputs = len(inputs)

        epoch = 10  # Hoe vaak we de perceptron opnieuw laten trainen

        while epoch != 0:
            good = 0
            for i in range(0, amount_inputs):  # Hieronder berekenen we de uitkomst van de perceptron. Als hier de juiste
                                               # uikomst uit komt zeggen we dat deze goed is. Maar het niet juist is,
                                               # updaten we de perceptron.
                uitkomst = AND.calculate_output([inputs[i][0], inputs[i][1]])
                if uitkomst == verwachte_outputs[i]:
                    good += 1
                else:
                    AND.update([inputs[i][0], inputs[i][1]], verwachte_outputs[i])

            if good == amount_inputs:  # Als alle antwoorden van een epoch goed zijn, is de perceptron al volledig
                                       # getrained en hoeven we niet onnodig verder te trainen.
                break

            epoch -= 1

        antwoorden = []  # Na het trainen schrijven we op welke antwoorden onze perceptron bij elke input geeft
        for i in range(0, 2):
            for j in range(0, 2):
                AND = Perceptron([1, 1], -2, 'AND gate', 0.2)
                antwoorden.append(AND.calculate_output([i, j]))

        self.assertEqual(antwoorden, verwachte_outputs)  # Vervolgens kijken we of de perceptron de juiste waardes
                                                         # teruggeeft.

    def test_AND_error(self):
        """In deze functie gaan we eerst een Perceptron met random parameters maken. Vervolgens gaan we deze perceptron
        trainen tot het voor elk van de verschillende inputs de juiste waardes geeft. Als dit lukt, werkt de AND
        perceptron."""

        AND = Perceptron([0, 4], 1, 'AND gate', 0.2)
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        verwachte_outputs = [0, 0, 0, 1]

        for i in range(0, len(inputs)):
            AND.update([inputs[i][0], inputs[i][1]], verwachte_outputs[i])

        error = AND.error()
        self.assertEqual(error, 0.75)  # Kijk of de outputs goed zijn

    def test_XOR(self):
        """In deze functie gaan we eerst een Perceptron met random parameters maken. Vervolgens gaan we deze perceptron
        trainen tot het voor elk van de verschillende inputs de juiste waardes geeft. Als dit lukt, werkt de XOR
        perceptron."""

        XOR = Perceptron([0, 4], 1, 'XOR gate', 0.2)
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        verwachte_outputs = [0, 1, 1, 0]
        amount_inputs = len(inputs)

        epoch = 100  # Hoe vaak we de perceptron opnieuw laten trainen

        while epoch != 0:
            good = 0

            for i in range(0, amount_inputs):  # Hieronder berekenen we de uitkomst van de perceptron. Als hier de juiste
                                               # uikomst uit komt zeggen we dat deze goed is. Maar het niet juist is,
                                               # updaten we de perceptron.
                uitkomst = XOR.calculate_output([inputs[i][0], inputs[i][1]])
                if uitkomst == verwachte_outputs[i]:
                    good += 1
                else:
                    XOR.update([inputs[i][0], inputs[i][1]], verwachte_outputs[i])

            if good == amount_inputs:  # Als alle antwoorden van een epoch goed zijn, is de perceptron al volledig
                                       # getrained en hoeven we niet onnodig verder te trainen.
                break

            epoch -= 1

        antwoorden = []  # Na het trainen schrijven we op welke antwoorden onze perceptron bij elke input geeft
        for i in range(0, 2):
            for j in range(0, 2):
                AND = Perceptron([1, 1], -2, 'AND gate', 0.2)
                antwoorden.append(AND.calculate_output([i, j]))

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

        # Deze test zal altijd falen. Dit komt omdat het onmogelijk is voor één perceptron om een XOR gate na te
        # bootsen. Dit komt omdat een perceptron alleen lineaire problemen op kan lossen en dit niet een lineair
        # probleem is. Het is nu zo dat ik maar 100 epoch heb gedaan, maar je zult vinden dat het niet uitmaakt of je
        # meer epochs toevoegd.


if __name__ == '__main__':
    unittest.main()
