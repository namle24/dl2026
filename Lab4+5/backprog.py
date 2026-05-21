import random

E = 2.718281828459045

def sigmoid(x):
    return 1 / (1 + E ** (-x))

def dsigmoid(x):
    return x * (1 - x)

class Neuron:

    def __init__(self, n_inputs):

        self.weights = []
        for i in range(n_inputs):
            self.weights.append(random.random())
        self.bias = random.random()
        self.output = 0
        self.delta = 0

class Layer:

    def __init__(self, n_neurons, n_inputs):

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

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
            new_outputs = []
            for neuron in layer.neurons:
                z = neuron.bias

                for i in range(len(outputs)):
                    z += outputs[i] * neuron.weights[i]

                neuron.output = sigmoid(z)
                new_outputs.append(neuron.output)
            outputs = new_outputs
        return outputs

    def backpropagation(self, inputs, target, lr):
        outputs = self.feedforward(inputs)
        #output layer 
        output_neuron = self.layers[-1].neurons[0]
        error = target - outputs[0]
        output_neuron.delta = error * dsigmoid(output_neuron.output)
        hidden_layer = self.layers[0]

        for i in range(len(hidden_layer.neurons)):
            neuron = hidden_layer.neurons[i]
            neuron.delta = (
                output_neuron.weights[i]
                * output_neuron.delta
                * dsigmoid(neuron.output)
            )

        hidden_outputs = []
        for neuron in hidden_layer.neurons:
            hidden_outputs.append(neuron.output)

        for i in range(len(output_neuron.weights)):
            output_neuron.weights[i] += (
                lr
                * output_neuron.delta
                * hidden_outputs[i]
            )

        output_neuron.bias += lr * output_neuron.delta
        # update hidden layer
        for neuron in hidden_layer.neurons:

            for i in range(len(neuron.weights)):
                neuron.weights[i] += (
                    lr
                    * neuron.delta
                    * inputs[i]
                )
            neuron.bias += lr * neuron.delta

        return error * error
 
    def load_data(self, filename):
        X = []
        Y = []
        with open(filename, "r") as file:
            lines = file.readlines()[1:]
            for line in lines:
                row = line.strip().split(",")
                X.append([
                    float(row[0]),
                    float(row[1])
                ])
                Y.append(float(row[2]))
        return X, Y

if __name__ == "__main__":
    nn = NeuralNetwork("network.txt")
    X, Y = nn.load_data("xor.csv")

    lr = 0.001
    threshold = 0.001
    epoch = 0
    total_loss = float('inf')
    
    while total_loss > threshold:
        total_loss = 0
        for i in range(len(X)):
            total_loss += nn.backpropagation(
                X[i],
                Y[i],
                lr
            )
        
        if epoch % 1000 == 0 and epoch > 0:
            print(f"Epoch {epoch}, Loss = {total_loss:.6f}")
        
        epoch += 1
        
    print(f"\nTraining completed after {epoch} epochs!")
    print(f"Final loss: {total_loss:.6f}")
    print(f"Threshold: {threshold}")
    
    print("\nFinal Results:")
    for i in range(len(X)):
        y = nn.feedforward(X[i])
        print(f"{X[i]} -> {y[0]:.6f}")
