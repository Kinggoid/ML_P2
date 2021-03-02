class Perceptron:
    errors = []

    def __init__(self, weights: list, bias: float, name: str, k: float):
        self.weights = weights
        self.name = name
        self.bias = bias
        self.learn_rate = k

    def calculate_output(self, inputs):
        """In deze definitie kijken we bij een perceptron welke inputs hij krijgt en hoe de weights en de bias dit
        beïnvloeden. Vervolgens kijken we of het antwoord de treshold haalt."""
        inputs_met_weight = [self.weights[i] * inputs[i] for i in range(0, len(inputs))]  # Inputs * weights
        if sum(inputs_met_weight) + self.bias >= 0:  # De som van die resultaten tellen we op met de bias en dan kijken we of het de treshold haalt.
            return 1
        return 0

    def update(self, inputs, target):
        """In deze functie gaan we kijken of en hoeveel we onze weights en biassen willen aanpassen."""
        p_error = target - self.calculate_output(inputs)  # Wat is de error
        self.errors.append(p_error)  # Voeg de error toe aan alle errors
        update = self.learn_rate * p_error
        self.bias = update + self.bias
        x = [update * i for i in inputs]
        self.weights = [x[i] + self.weights[i] for i in range(0, len(inputs))]

    def error(self, error):
        """Berekenen de totale error van alle trainingvoorbeelden."""
        return (sum(error) ** 2) / len(self.errors)

    def __str__(self):
        """Informatie van de perceptron"""
        return 'Mijn naam is {} en ik heb {} input variabelen. Mijn bias is {}.'.format(self.name, str(len(self.weights)), str(self.bias))



