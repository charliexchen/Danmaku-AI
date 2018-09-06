import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x, scale=0.2):
    if x > 0:
        return x
    else:
        return scale * x


def tanh(x, scale=1.0):
    return scale * np.tanh(x)


class dense_layer():
    def __init__(self, input_size, output_size, activation, initvar=0.2):
        self.initvar = initvar
        self.weights = np.random.normal(0, initvar, [output_size, input_size + 1])
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

    def evaluate(self, input):
        try:
            assert (len(input) == self.input_size)
        except AssertionError:
            print("Input size incorrect: expected {} but obtained {}".format(self.input_size, len(input)))
        return np.array([self.activation(row[0] + sum(input * row[1:])) for row in self.weights])

    def mutate(self, rate):
        self.weights += np.random.normal(0, rate, [self.output_size, self.input_size + 1])

    def reset(self, var=0.2):
        self.weights = np.random.normal(0, var, [self.output_size, self.input_size + 1])


class dense_net():
    def __init__(self, input_size, output_size, output_rule, initvar=0.2, recursive=False):
        self.input_size = input_size
        self.layers = [dense_layer(input_size, output_size, output_rule, initvar)]
        self.recursive = recursive
        if self.recursive:
            self.previous = [0 for i in range(output_size)]

    def add_layer(self, output_size, activation, initvar=0.2):
        self.layers.append(dense_layer(self.layers[-1].output_size, output_size, activation, initvar))
        if self.recursive:
            self.layers[0] = dense_layer(self.layers[-1].output_size + self.input_size, self.layers[0].output_size,
                                         self.layers[0].activation, self.layers[0].initvar)
            self.previous = [0 for i in range(output_size)]

    def evaluate(self, input):
        if self.recursive:
            input = list(input) + list(self.previous)
        try:
            if self.recursive:
                assert (len(input) == self.input_size+self.layers[-1].output_size)
            else:
                assert (len(input) == self.input_size)
        except AssertionError:
            print("Input size incorrect: expected {} but obtained {}".format(self.input_size, len(input)))
        output = input
        for layer in self.layers:
            output = layer.evaluate(output)
        self.previous = output
        return output

    def mutate(self, rate):
        if type(rate) == float:
            for layer in self.layers:
                layer.mutate(rate)
        else:
            assert (len(self.layers) == len(rate))
            for i in range(len(self.layers)):
                layer.mutate(rate[i])
    def reset_rec(self, ):
        try:
            assert(self.recursive==True)
        except AssertionError:
            print("Error -- tried to reset recursion inputs to non-recursive net")
        self.previous = [0 for i in self.previous]

    def reset(self, rate):
        if type(rate) == float:
            for layer in self.layers:
                layer.reset(rate)
        else:
            assert (len(layers) == len(rate))
            for i in range(len(self.layers)):
                layer.reset(rate[i])


if __name__ == "__main__":
    net = dense_net(3, 10, tanh)
    print(net.layers[0].weights)
    net.add_layer(2, tanh)
    print(net.evaluate(np.array([1, 1, 1])))
    net.mutate(0.05)
    print(net.evaluate(np.array([1, 1, 1])))


    def timetest():
        net = dense_net(3, 10, tanh)
        net.add_layer(2, tanh)
        for i in range(1000):
            net.evaluate(np.array([1, 1, 1]))

    import cProfile

    cProfile.run("timetest()")
