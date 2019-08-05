from Learning.neural_net import relu, tanh
import numpy as np
from copy import deepcopy
import random


def identity(x):
    return x


def xor(input):
    x, y = input
    return 1 if x != y else 0


def fitness_xor(net):
    inputs = [[1, 1], [0, 1], [1, 0], [0, 0]]
    net.fitness = -sum(
        [(net.eager_activate(input)[0] - xor(input)) ** 2 for input in inputs]
    ) + np.random.normal(0, 0.001)


class DAG_net:
    def __init__(self, input_size, output_size, nodes, edges):
        # precompute a the topological ordering
        pass


class NodeType:
    BIAS = "bias"
    INPUT = "input"
    OUTPUT = "output"
    HIDDEN = "hidden"
    STATE = "state"


class NEATNode:
    def __init__(self, params):
        # this data_structure holds the information in each node

        # Index is the innovation number of the node
        assert (
            "index" in params
        ), 'Missing inputs -- "index" is required for the node params'
        # node_type can be input/output/bias/hidden
        assert (
            "node_type" in params
        ), 'Missing inputs -- "node_type" is required for the node params'
        self.index = params["index"]
        self.node_type = params["node_type"]

        if self.node_type == NodeType.INPUT or self.node_type == NodeType.OUTPUT:
            # The IO_index is the index of the nodes in input/output vector
            assert (
                "IO_index" in params
            ), "Missing inputs - IO_index required for {} nodes".format(self.node_type)
            self.IO_index = params["IO_index"]

        if self.node_type == NodeType.INPUT or self.node_type == NodeType.BIAS:
            assert (
                "activation" not in params
            ), "Error -- {} nodes should not have activation functions".format(
                self.node_type
            )
        else:
            # Denotes the activation function for the node
            assert (
                "activation" in params
            ), "Error -- No activation function found for {} node".format(
                self.node_type
            )
            self.activation = params["activation"]
        self.complete = False
        self.val = 0

    def reset(self):
        self.complete = False
        self.val = 0

    def compute(self):
        self.val = self.activation(self.val)
        self.complete = True


class NEATEdge:
    def __init__(self, parent_ind, child_ind, innovation_num, weight, active=True):
        self.parent_ind = parent_ind
        self.child_ind = child_ind
        self.innovation_num = innovation_num
        self.weight = weight
        self.active = active

    def __repr__(self, detailed=False):
        if self.active:
            if detailed:
                return "{}th node {}->{} with weight {}".format(
                    self.innov, self.parent_ind, self.child_ind, self.weight
                )
            else:
                return "{}->{}".format(self.parent_ind, self.child_ind)
        return "{}->{}(dis)".format(self.parent_ind, self.child_ind)


class GlobalInnovations:
    def __init__(self):
        # edge is a tuple (node1 innovation number, node2 innovation number)
        self.edge_to_innovs = {}  # edges may have multiple innovations, some disabled.
        self.innov_to_edge = {}
        self.edge_innovs = 0

        # map from node innovation number to the edge innovation it disabled at its creation
        self.node_to_disabled = {}
        self.disabled_to_node = {}
        self.node_innovs = 0

    def add_edge_if_new(self, edge):
        if edge not in self.edge_to_innov:
            new_edge_innov = self.edge_innovs
            self.edge_to_innov[edge] = {new_edge_innov}
            self.innov_to_edge[new_edge_innov] = edge
            self.edge_innovs += 1
            return {new_edge_innov}
        else:
            self.edge_to_innov[edge].add(self.edge_innovs)
            self.edge_innovs += 1
            return self.edge_to_innov[edge]

    def add_node_if_new(self, innov):
        if innov not in self.disabled_to_node:
            new_node_innov = self.node_innovs
            self.node_to_disabled[new_node_innov] = innov
            self.disabled_to_node[innov] = new_node_innov
            self.node_innovs += 1
            return new_node_innov
        else:
            return self.disabled_to_node(innov)


class Genome:
    def __init__(self):
        self.edge_genome = []
        self.node = []

    def add_edge(self):
        pass

    def add_node(self, edge):
        self.edge_genome


class NEATPopulation:
    def __init__(self, input_size, output_size, output_activation, population_size):
        self.innovations = GlobalInnovations
        self.population_size = population_size
        self.population = [
            NEATNet(input_size, output_size, output_activation)
            for _ in range(population_size)
        ]

    def find_fitness(self, fitness_function):
        for net in self.population:
            fitness_function(net)

    def breed(self, scale):
        new_population = self.population
        count = 0
        survivor_count = len(self.population)
        while len(new_population) < self.population_size:
            new_net = deepcopy(self.population[count % survivor_count])
            new_net.mutate(scale)
            new_population.append(new_net)
        self.population = new_population

    def cull(self, survivors):
        random.shuffle(self.population)
        list.sort(self.population, key=lambda x: -x.fitness)
        self.population = self.population[:survivors]


class NEATNet:
    def __init__(self, input_size, output_size, output_activation):
        self.input_size = input_size
        self.output_size = output_size
        self.fitness = 0

        # This dictionary stores the map from innovation numbers to the edges
        # This represents the genotype of this net
        self.edge_genome = {}

        # This dictionary of dictionaries stores the sparse adjacency matrix of the net.
        # The values are the innovation numbers of the edges
        self.adjacency_matrix = {}

        # This is the node genome - we initialise it with the bias node
        bias_params = {"index": 0, "node_type": NodeType.BIAS}
        self.node_genome = [NEATNode(bias_params)]

        # This keeps track of which nodes are the outputs
        self.output_index = []

        # We use this to initialise the innovation numbers for the I/O nodes
        i = 1
        for j in range(input_size):
            input_param = {"index": i, "node_type": NodeType.INPUT, "IO_index": j}
            self.node_genome.append(NEATNode(input_param))
            i += 1
        for j in range(output_size):
            self.output_index.append(i)
            self.adjacency_matrix[i] = {}
            output_param = {
                "index": i,
                "node_type": NodeType.OUTPUT,
                "IO_index": j,
                "activation": output_activation,
            }
            self.node_genome.append(NEATNode(output_param))
            i += 1

    def find_path(self, node1, node2):
        # Check if there is a path from node 1 to node 2
        # We need to ensure that our neural net is a directed acyclic graph
        # Hence we need this function to see if an additional path will create a cycle
        # As convention, the input and bias nodes will have no parents and the output nodes will have no children
        if node2 not in self.adjacency_matrix and node1 != node2:
            return False
        nodes = set([node2])
        reached = set([node2])
        target = node1
        while nodes:
            if target in nodes:
                return True
            new_nodes = set([])
            for node in nodes:
                if node in self.adjacency_matrix:
                    for adj in self.adjacency_matrix[node].keys():
                        if adj not in reached:
                            reached.add(adj)
                            new_nodes.add(adj)
            nodes = new_nodes
        return False

    def valid_edge(self, node1, node2):
        # node1 -> node2
        if node1 == node2:
            return False
        # Check if the edge already exists
        if node2 in self.adjacency_matrix:
            if node1 in self.adjacency_matrix[node2]:
                return False
        # Check if adding edge node1 -> node2 will ruin anything
        n1 = self.node_genome[node1].node_type
        n2 = self.node_genome[node2].node_type
        # Bias nodes are basically input nodes which always output 1. This simplifies the number of conditionals later
        if n1 == NodeType.BIAS:
            n1 = NodeType.INPUT
        if n2 == NodeType.BIAS:
            n2 = NodeType.INPUT

        # make sure that the output node is not a parent
        if n1 == NodeType.OUTPUT:
            return False
        # make sure that the input node is not a child
        if n2 == NodeType.INPUT:
            return False
        if n1 == n2 and n2 != NodeType.HIDDEN:
            # We cannot link inputs to inputs or hidden to hidden
            return False

        # if there is a path from node2 -> node1, then node 1 -> node 2 will result in a cycle
        return not self.find_path(node2, node1)

    def add_edge(self, node1, node2, innov, weight):
        if self.edge_genome:
            innov = max(max(self.edge_genome.keys()) + 1, innov)
        if self.valid_edge(node1, node2):
            # adds edge node1 -> node2
            if node2 not in self.adjacency_matrix:
                self.adjacency_matrix[node2] = {}
            self.adjacency_matrix[node2][node1] = innov
            self.edge_genome[innov] = NEATEdge(node1, node2, innov, weight)
            return True
        return False

    def add_random(self):
        pass

    def edge_weight(self, child, parent):
        edge = self.edge_genome[self.adjacency_matrix[child][parent]]
        if edge.active:
            return edge.weight
        else:
            return 0

    def add_node(self, node1, node2, innov=0, activation=tanh):
        if (
            node2 in self.adjacency_matrix
            and node1 in self.adjacency_matrix[node2]
            and self.edge_weight(node2, node1) != 0
        ):
            # We add a new node if there is an activated edge between the two
            # Create new node
            new_index = len(self.node_genome)
            hidden_param = {
                "index": new_index,
                "node_type": NodeType.HIDDEN,
                "activation": activation,
            }
            self.node_genome.append(NEATNode(hidden_param))

            # We want to minimise the the effect of the mutation, so we simply use the same weight after the new node and weight 1 before
            old_edge = self.adjacency_matrix[node2][node1]
            old_val = self.edge_genome[old_edge].weight
            # We add a new edge
            self.add_edge(new_index, node2, innov, old_val)
            self.add_edge(node1, new_index, innov + 1, 1)
            # Disable the edge which used to be between the two
            self.edge_genome[old_edge].active = False

            # Return true if the node was indeed added
            return True
        return False

    def eager_activate(self, input):
        # Remove the values in all the nodes
        for node in self.node_genome:
            node.reset()

        # Dynamic programming memory -- the keys are the innovation position of the edges, the values are sets of parents of each node
        # We remove elements from the values as we calculate them
        mem = {}
        # Stack in order to implement the function without recursion
        stack = []

        # Initialise the output nodes in the stack
        for i in self.output_index:
            stack.append([i, set(self.adjacency_matrix[i].keys())])
        # Loop while the stack is still non-empty
        while stack:
            if len(stack[-1][1]) == 0:
                # If all the values in the parents of the last node in the stack have been calculated and summed, remove from the stack and apply the activation function
                node = self.node_genome[stack[-1][0]]
                node.compute()
                mem[stack[-1][0]] = node.val
                stack.pop()
            else:
                # If there are still values which have not been summed to the last node in the stack we want to either calculate the the hidden value or
                par = stack[-1][1].pop()
                chi = stack[-1][0]
                chi_node = self.node_genome[chi]
                par_node = self.node_genome[par]
                if par in mem:
                    # If the node is hidden (i.e. might have parents) but have been calculated, we multiply the node value by the edge
                    chi_node.val += mem[par] * self.edge_weight(chi, par)
                elif self.node_genome[par].node_type == NodeType.BIAS:
                    # If the node is the bias node, we simply add the weight
                    chi_node.val += self.edge_weight(chi, par)
                elif self.node_genome[par].node_type == NodeType.INPUT:
                    # If the node is an input node, we multiply the input value by the edge
                    chi_node.val += input[par_node.IO_index] * self.edge_weight(
                        chi, par
                    )
                else:
                    # For hidden nodes, we need to calculate the node value value if it's not already calculated
                    stack[-1][1].add(par)
                    stack.append([par, set(self.adjacency_matrix[par].keys())])
        return [self.node_genome[i].val for i in self.output_index]

    def mutate_weights(self, scale=0.1):
        for edge in self.edge_genome:
            self.edge_genome[edge].weight += np.random.normal(0, scale)

    def mutate(self, scale, edge_mute_prob=0.5, node_mute_prob=0.5):
        # mutate the weights
        self.mutate_weights(scale)

        # mutate the topology of the net
        if np.random.random() < edge_mute_prob:
            not_added = True
            allowed_attempts = 10
            while not_added and allowed_attempts > 0:
                allowed_attempts -= 1
                chi = np.random.randint(0, len(self.node_genome))
                par = np.random.randint(0, len(self.node_genome))
                not_added = not self.add_edge(par, chi, 0, 0)
        if np.random.random() < node_mute_prob:
            chi = np.random.randint(0, len(self.node_genome))
            par = np.random.randint(0, len(self.node_genome))
            self.add_node(par, chi, activation=tanh)

    @staticmethod
    def breed(net1, net2, equal_threshold):
        fitness_diff = net1.fitness - net2.fitness
        if abs(fitness_diff) < equal_threshold:
            gene1 = net1.edge_genome
            gene2 = net2.edge_genome

            innovations = gene1.keys().union(gene2.keys())

            new_genome = {}

            for innovation in innovations:
                if innovation in gene1 and innovation in gene2:
                    new_genome[innovation] = deepcopy(gene1[innovation])

        if fitness_diff < 0:
            # without loss of generality, fitness of net1 is higher
            bkup = net1
            net1 = net2
            net2 = bkup


def test_neat_basic_function():
    def almost_equal(a, b, thres=0.00000001):
        return np.linalg.norm(np.array(a) - np.array(b)) < thres

    # A test function to ensure that the stuff in the Neat module works as intended

    # create a new thing
    neat = NEATNet(2, 2, identity)
    neat.add_edge(0, 3, 0, weight=0.1)
    neat.add_edge(1, 3, 0, weight=0.2)
    neat.add_edge(1, 4, 0, weight=-0.1)
    neat.add_node(1, 4)
    neat.add_edge(2, 5, 0, weight=-0.1)
    assert almost_equal(neat.eager_activate([0, 0]), [0.1, 0.0])
    assert almost_equal(neat.eager_activate([1, 0]), [0.3, -0.1])
    assert almost_equal(neat.eager_activate([-1, 0]), [-0.1, 0.02])
    assert almost_equal(neat.eager_activate([1, 1]), [0.3, -0.09])
    print("Basic NEAT test passed")


def timetest():
    neat = NEATNet(1, 1, identity)
    for i in range(100):
        neat.mutate(0.2)
        print(neat.edge_genome)


def mutation_test():
    neat = NEATNet(2, 1, identity)
    for i in range(50):
        neat.mutate(0.1)
    print([gene for gene in neat.edge_genome.values() if gene.active])


def xor_evolution_test():
    pop = NEATPopulation(2, 1, identity, 250)
    while True:
        pop.find_fitness(fitness_xor)
        pop.cull(10)
        pop.breed(0.1)
        print(max(net.fitness for net in pop.population))
        print(max(len(net.edge_genome) for net in pop.population))


if __name__ == "__main__":
    xor_evolution_test()
    neat = NEATNet(2, 1, identity)
    neat.add_edge(0, 3, 0, weight=0.1)
    neat.add_edge(1, 3, 0, weight=0.2)
