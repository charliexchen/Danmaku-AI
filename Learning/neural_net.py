import numpy as np


def relu(x, scale=0.2):
    if x > 0:
        return x
    else:
        return scale * x


def d_relu(x, scale=0.2):
    if x > 0:
        return 1
    else:
        return scale


def tanh(x, scale=1.0):
    return scale * np.tanh(x)


def dtanh(x, scale=1.0):
    return scale * (np.sech(x) ** 2)


class dense_layer:
    def __init__(self, input_size, output_size, activation, initvar=0.2):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.initvar = initvar
        self.reset(initvar)

    def activate(self, input):
        assert (
            len(input) == self.input_size
        ), "Input size incorrect: expected {} but obtained {}".format(
            self.input_size, len(input)
        )
        self.output = np.array(
            [self.activation(row[0] + sum(input * row[1:])) for row in self.weights]
        )
        return self.output

    def mutate(self, rate):
        self.weights += np.random.normal(
            0, rate, [self.output_size, self.input_size + 1]
        )

    def reset(self, var=0.2):
        self.weights = np.random.normal(0, var, [self.output_size, self.input_size + 1])

    @staticmethod
    def export_weights(layer):
        params = {"activation": layer.activation, "weights": layer.weights}
        try:
            params["var"] = layer.initvar
        except AttributeError:
            pass
        return params

    @staticmethod
    def import_weights(params):
        weights = params["weights"]
        output_size = len(weights)
        assert output_size > 0, "Error - Output size is 0"
        input_size = len(weights[0]) - 1
        assert input_size > 0, "Error - Input size is 0"
        activation = params["activation"]
        if "initvar" in params:
            var = params["initvar"]
        else:
            var = 0.2
        layer = dense_layer(input_size, output_size, activation, var)
        layer.weights = weights
        return layer


class dense_net:
    def __init__(
        self,
        input_size,
        output_size,
        activation,
        initvar=0.2,
        recursive=False,
        state_size=4,
    ):
        self.input_size = input_size
        self.layers = [dense_layer(input_size, output_size, activation, initvar)]
        self.recursive = recursive
        if self.recursive:
            self.state_size = state_size
            self.previous = [0 for i in range(state_size)]
            self.finalised = False

    def add_layer(self, output_size, activation, initvar=0.2, final=False):
        if self.recursive:
            if final:
                try:
                    assert not self.finalised
                except AssertionError:
                    print("Error -- Rec Net already finalised")
                self.layers.append(
                    dense_layer(
                        self.layers[-1].output_size,
                        output_size + self.state_size,
                        activation,
                        initvar,
                    )
                )
                self.layers[0] = dense_layer(
                    self.state_size + self.input_size,
                    self.layers[0].output_size,
                    self.layers[0].activation,
                    self.layers[0].initvar,
                )
                self.previous = [0 for i in range(self.state_size)]
                self.finalised = True
            else:
                self.layers.append(
                    dense_layer(
                        self.layers[-1].output_size, output_size, activation, initvar
                    )
                )
        else:
            self.layers.append(
                dense_layer(
                    self.layers[-1].output_size, output_size, activation, initvar
                )
            )

    def activate(self, input):
        if self.recursive:
            assert self.finalised, "Error -- Evaluating Rec net before finalisation"
            input = list(input) + list(self.previous)

        if self.recursive:
            assert (
                len(input) == self.input_size + self.state_size
            ), "Input size incorrect: expected {} but obtained {}".format(
                self.input_size + self.state_size, len(input)
            )
        else:
            assert (
                len(input) == self.input_size
            ), "Input size incorrect: expected {} but obtained {}".format(
                self.input_size, len(input)
            )

        output = input
        for layer in self.layers:
            output = layer.activate(output)

        if self.recursive:
            output_length = self.layers[-1].output_size - self.state_size
            self.previous = output[output_length:]
            return output[:output_length]
        else:
            return output

    def mutate(self, rate):
        if type(rate) == float:
            for layer in self.layers:
                layer.mutate(rate)
        else:
            assert len(self.layers) == len(rate)
            for i in range(len(self.layers)):
                self.layers[i].mutate(rate[i])

    def reset_rec(self):
        assert (
            self.recursive == True
        ), "Error -- tried to reset recursion inputs to non-recursive net"
        self.previous = [0 for i in self.previous]

    def reset(self, rate):
        if type(rate) == float:
            for layer in self.layers:
                layer.reset(rate)
        else:
            assert len(self.layers) == len(rate)
            for i in range(len(self.layers)):
                self.layers[i].reset(rate[i])

    @staticmethod
    def export_weights(net):
        weights = [dense_layer.export_weights(layer) for layer in net.layers]
        output = {"weights": weights}
        if net.recursive:
            output["state_size"] = net.rec_size
        return output

    @staticmethod
    def import_weights(params):
        layers = [dense_layer.import_weights(weight) for weight in params["weights"]]
        input_size = layers[0].input_size
        output_size = layers[0].output_size
        activation = layers[0].activation
        initvar = layers[0].initvar
        if "state_size" in params:
            state_size = params["state_size"]
            net = dense_net(
                input_size - state_size,
                output_size,
                activation,
                initvar,
                True,
                state_size,
            )
            net.finalised = True
        else:
            net = dense_net(input_size, output_size, activation, initvar, False)

        net.layers = layers
        return net


class diff_dense_net(dense_net):
    def __init__(
        self,
        input_size,
        output_size,
        output_rule,
        initvar=0.2,
        recursive=False,
        rec_size=4,
    ):
        super(diff_dense_net, self).__init__(
            input_size, output_size, output_rule, initvar, recursive, rec_size
        )


def LSTM():
    def __init__(self, nets, input_size, latent_size, output_size):
        self.output_size = output_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.nets = nets
        self.latent1 = [0 for i in range(latent_size)]
        self.latent2 = [0 for i in range(latent_size)]
        assert len(nets) == 5, "Error - LSTM requires 5 nets"
        try:
            for i in range(4):
                assert net[i].input_size == input_size + latent_size
                assert nets[i].output_size == latent_size
                if i != 2:
                    assert nets[i].layers[-1].activation == sigmoid
                else:
                    assert nets[i].layers[-1].activation == tanh
            assert nets[4].output_size == output_size
        except:
            print("Error - component neural nets wrong size")

    def activate(self, input):
        concat_input = input + self.latent1
        concat_output = [net.activate(concat_input) for net in self.nets[:3]]
        scale1 = [
            self.latent2[i] * concat_output[0][i] for i in range(self.latent_size)
        ]
        scale2 = [
            concat_output[2][i] * concat_output[1][i] for i in range(self.latent_size)
        ]
        self.latent2 = [tanh(scale1[i] + scale2[i]) for i in range(self.latent_size)]
        output = [self.latent2[i] * concat_output[3] for i in range(self.latent_size)]
        self.latent1 = output

        return self.nets[4].activate(output)

    def mutate(self, rate):
        for net in self.nets:
            net.mutate(rate)


def make_focused(net, sensor_len):
    net.input_size += 2
    net.layers[0].input_size += 2
    net.layers[0].weights = np.insert(
        net.layers[0].weights, sensor_len + 1, 0.0, axis=1
    )
    net.layers[0].weights = np.insert(
        net.layers[0].weights, sensor_len + 1, 0.0, axis=1
    )
    output_size = net.layers[-1].output_size - net.rec_size
    net.layers[-1].output_size += 1
    net.layers[-1].weights = np.insert(net.layers[-1].weights, output_size, 0.0, axis=0)
    return net


if __name__ == "__main__":
    net = dense_net(3, 10, tanh)
    print(net.layers[0].weights)
    net.add_layer(2, tanh)
    print(net.activate(np.array([1, 1, 1])))
    net.mutate(0.05)
    print(net.activate(np.array([1, 1, 1])))

    def timetest():
        net = dense_net(3, 10, tanh)
        net.add_layer(2, tanh)
        for i in range(1000):
            net.activate(np.array([1, 1, 1]))

    import cProfile

    cProfile.run("timetest()")
