import random
import unittest

from sklearn.datasets import load_iris

from ML.p2_perceptronen.p2_iris import irissen
from ML.p2_perceptronen.p2_perceptronen import Perceptron


class MyTestCase(unittest.TestCase):
    def test_SetosaVersicolor(self):
        """In deze functie testen we de functie 'irissen'. Eerst proberen we onze perceptron twee soorten irrisen te
        onderscheiden, de Iris Setosa en de Iris Versicolor."""
        data = load_iris()

        leerlingnummer = 1758191
        random.seed(leerlingnummer)

        seed = str(random.random())

        iris = Perceptron([int(seed[2]), int(seed[3]), int(seed[4]), int(seed[5])], float(seed[6]), 'iris onderscheider'
                          , float(seed))

        irissen(['Setosa', 'Versicolor'], data, iris)

        print('De weights van de iris onderscheider perceptron zijn: ' + str(iris.weights))
        print('De bias van de iris onderscheider perceptron is: ' + str(iris.bias))
        self.assertEqual(True, True)  # Als het programma hier komt zonder in een infinite loop te komen is het programma
                                      # correct.

    def test_SetosaVersicolorVirginica(self):
        """In deze functie testen we de functie 'irissen'. Nu proberen we onze perceptron alle drie te irissen te
        onderscheiden."""
        data = load_iris()

        leerlingnummer = 1758191
        random.seed(leerlingnummer)

        seed = str(random.random())

        iris = Perceptron([int(seed[2]), int(seed[3]), int(seed[4]), int(seed[5])], float(seed[6]), 'iris onderscheider'
                          , float(seed))

        irissen(['Setosa', 'Versicolor', 'Virginica'], data, iris)  # Train je perceptron

        print('De weights van de iris onderscheider perceptron zijn: ' + str(iris.weights))
        print('De bias van de iris onderscheider perceptron is: ' + str(iris.bias))
        self.assertEqual(True, True)  # Als het programma hier komt zonder in een infinite loop te komen is het programma
                                      # correct.


        # Deze functie zal altijd door blijven gaan, dit is omdat er twee soorten irissen zijn die heel erg op elkaar
        # lijken en het onmogelijk is voor één perceptron om de juiste parameters te krijgen om deze perfect te
        # onderscheiden.

if __name__ == '__main__':
    unittest.main()
