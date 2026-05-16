import random

E = 2.718281828459045


def sigmoid(x):
    return 1 / (1 + E ** (-x))

def threshold(x):
    return 1 if x >= 0 else 0

class Neuron:
    def __init__(self, n_inputs):
        self.weights = []
        self.bias = 0
        for i in range(n_inputs):
            self.weights.append(random.random())
        self.bias = random.random()
    
    def feedforward(self, inputs):
        z = self.bias
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        return threshold(z)


class Layer:
    def __init__(self, n_neurons, n_inputs):
        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

    def feedforward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feedforward(inputs))
        return outputs


class NeuralNetwork:
    def __init__(self, filename):
        self.layers = []
        with open(filename, "r") as file:
            lines = file.readlines()
            N = int(lines[0])
            sizes = []
            for i in range(1, N + 1):
                sizes.append(int(lines[i]))

        for i in range(1, len(sizes)):
            self.layers.append(
                Layer(sizes[i], sizes[i - 1])
            )

    def feedforward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.feedforward(outputs)
        return outputs


if __name__ == "__main__":
    nn = NeuralNetwork("network.txt")
    # hidden layer
    nn.layers[0].neurons[0].weights = [-1.0, -1.0]
    nn.layers[0].neurons[0].bias = 1.5

    nn.layers[0].neurons[1].weights = [1.0, 1.0]
    nn.layers[0].neurons[1].bias = -0.5

    # output layer
    nn.layers[1].neurons[0].weights = [1.0, 1.0]
    nn.layers[1].neurons[0].bias = -1.5

    test_data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    for x in test_data:
        y = nn.feedforward(x)
        print(x, "->", y[0])
