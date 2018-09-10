from neural_net import relu, sigmoid, tanh
import numpy as np
import pdb
import cProfile


def identity(x):
    return x


class neat_node:
    def __init__(self, params):
        # this datastructure holds the information in each node
        # params = kwargs.items()
        try:
            # Index is the innovation number of the node
            assert "index" in params
            # node_type can be input/output/bias/hidden
            assert "node_type" in params
            self.index = params["index"]
            self.node_type = params["node_type"]
        except AssertionError:
            print("Missing inputs -- index and node_type required for neat_node")

        if self.node_type == "input" or self.node_type == "output":
            try:
                # The IO_index is the index of the nodes in input/output vector
                assert "IO_index" in params
                self.IO_index = params["IO_index"]
            except AssertionError:
                print("Missing inputs - IO_index required for {} nodes".format(self.node_type))

        if self.node_type == "input" or self.node_type == "bias":
            try:
                assert "activation" not in params
            except AssertionError:
                if self.node_type != "input":
                    print("Error -- {} nodes should not have activation functions".format(self.node_type))
        else:
            try:
                # Denotes the activation function for the node
                assert "activation" in params
                self.activation = params["activation"]
            except AssertionError:
                print("Error -- No activation function found for {} node".format(self.node_type))
        self.complete = False
        self.val = 0

    def reset(self):
        self.complete = False
        self.val = 0

    def compute(self):
        self.val = self.activation(self.val)
        self.complete = True


class neat_edge:
    def __init__(self, par_index, chi_index, innov, val, active=True):
        self.par_index = par_index
        self.chi_index = chi_index
        self.val = val
        self.active = active
        self.innov = innov

    def __repr__(self):
        if self.active:
            return "{} -> {}".format(self.par_index, self.chi_index)
        return "{} -> {} (disabled)".format(self.par_index, self.chi_index)
        return "{}th node {} -> {} with weight {}".format(self.innov, self.par_index, self.chi_index, self.val)


class Neat:
    def __init__(self, input_size, output_size, output_activation):

        self.input_size = input_size
        self.output_size = output_size
        self.fitness = 0

        # Three datastructures to store the edge genome
        # This dictionary stores the map from innovation numbers to the edges
        self.edge_innov = {}
        # This dictionary of dictionaries stores the map from adjacency to the innovation numbers
        self.edge_adj = {}

        # This is the node genome -- we initialise it with the bias node
        bias_params = {"index": 0, "node_type": "bias"}
        self.nodes = [neat_node(bias_params)]

        # We need this for evaluating the function later
        self.output_index = []

        # We use this to initialise the innovation numbers for the I/O nodes
        dummy_index = 1
        for i in range(input_size):
            input_param = {"index": dummy_index, "node_type": "input", "IO_index": i}
            self.nodes.append(neat_node(input_param))
            dummy_index += 1
        for i in range(output_size):
            self.output_index.append(dummy_index)
            self.edge_adj[dummy_index] = {}
            output_param = {"index": dummy_index, "node_type": "output", "IO_index": i, "activation": output_activation}
            self.nodes.append(neat_node(output_param))
            dummy_index += 1

    def find_path(self, node1, node2):
        # Check if there is a path from node 1 to node 2
        # We need to ensure that our neural net is a directed acyclic graph
        # Hence we require this function to see if an additional path will create a cycle
        # Besides, it's good practice
        # As convention, the input and bias nodes will have no parents and the output nodes will have no children
        if node2 not in self.edge_adj and node1 != node2:
            return False
        nodes = set([node2])
        reached = set([node2])
        target = node1
        while nodes:
            if target in nodes:
                return True
            new_nodes = set([])
            for node in self.edge_adj:
                for adj in self.edge_adj[node]:
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
        if node2 in self.edge_adj:
            if node1 in self.edge_adj[node2]:
                return False
        # Check if adding edge node1 -> node2 will ruin anything
        n1 = self.nodes[node1].node_type
        n2 = self.nodes[node2].node_type
        # Bias nodes are basically input nodes which always output 1. This simplifies the number of conditionals later
        if n1 == "bias":
            n1 = "input"
        if n2 == "bias":
            n2 = "input"

        # make sure that the output node is not a parent
        if n1 == "output":
            return False
        # make sure that the input node is not a child
        if n2 == "input":
            return False
        if n1 == n2 and n2 != "hidden":
            # We cannot link inputs to inputs or hidden to hidden
            return False

        # if there is a path from node2 -> node1, then node 1 -> node 2 will result in a cycle
        return not self.find_path(node2, node1)

    def add_edge(self, node1, node2, innov, val):
        if self.edge_innov:
            #pdb.set_trace()
            innov = max(max(self.edge_innov.keys()) + 1, innov)
        if self.valid_edge(node1, node2):
            # adds edge node1 -> node2
            if node2 not in self.edge_adj:
                self.edge_adj[node2] = {}
            self.edge_adj[node2][node1] = innov
            self.edge_innov[innov] = neat_edge(node1, node2, innov, val)

            return True
        return False

    def add_random(self):
        pass

    def edge_weight(self, child, parent):
        edge = self.edge_innov[self.edge_adj[child][parent]]
        if edge.active:
            return edge.val
        else:
            return 0

    def add_node(self, node1, node2, innov=0, activation=relu):
        if node2 in self.edge_adj and node1 in self.edge_adj[node2] and self.edge_weight(node2, node1) != 0:
            # We add a new node if there is an activated edge between the two
            # Create new node
            new_index = len(self.nodes)
            hidden_param = {"index": new_index, "node_type": "hidden", "activation": activation}
            self.nodes.append(neat_node(hidden_param))

            # We want to minimise the the effect of the mutation, so we simply use the same weight after the new node and weight 1 before
            old_edge = self.edge_adj[node2][node1]
            old_val = self.edge_innov[old_edge].val
            # We add a new edge
            self.add_edge(new_index, node2, innov, old_val)
            self.add_edge(node1, new_index, innov, 1)

            # Disable the edge which used to be between the two
            self.edge_innov[old_edge].active = False
            # Return true if the node was indeed added
            return True
        return False

    def activate(self, input):
        # Remove the values in all the nodes
        for node in self.nodes:
            node.reset()
        # Dynamic programming memory -- the keys are the innovation position of the edges, the values are sets of parents of each node
        # We remove elements from the values as we calculate them
        mem = {}
        # Stack in order to implement the function without recursion
        stack = []
        # Initialise the output nodes in the stack
        for i in self.output_index:
            stack.append([i, set(self.edge_adj[i].keys())])
        # Loop while the stack is still non-empty
        while stack:
            if len(stack[-1][1]) == 0:
                # If all the values in the parents of the last node in the stack have been calculated and summed, remove from the stack and apply the activation function
                node = self.nodes[stack[-1][0]]
                node.compute()
                mem[stack[-1][0]] = node.val
                stack.pop()
            else:
                # If there are still values which have not been summed to the last node in the stack we want to either calculate the the hidden value or
                par = stack[-1][1].pop()
                chi = stack[-1][0]
                chi_node = self.nodes[chi]
                par_node = self.nodes[par]
                # if chi==3:
                # pdb.set_trace()
                if par in mem:
                    # If the node is hidden (i.e. might have parents) but have been calculated, we multiply the node value by the edge
                    chi_node.val += mem[par] * self.edge_weight(chi, par)
                elif self.nodes[par].node_type == "bias":
                    # If the node is the bias node, we simply add the weight
                    chi_node.val += self.edge_weight(chi, par)
                elif self.nodes[par].node_type == "input":
                    # If the node is an input node, we multiply the input value by the edge
                    # pdb.set_trace()
                    chi_node.val += input[par_node.IO_index] * self.edge_weight(chi, par)
                else:
                    # For hidden nodes, we need to calculate the node value value if it's not already calculated
                    stack[-1][1].add(par)
                    stack.append([par, set(self.edge_adj[par].keys())])
        return [self.nodes[i].val for i in self.output_index]

    def export_simplified(self):
        pass

    def mutate_weights(self, scale=0.2):
        for edge in self.edge_innov:
            self.edge_innov[edge].val += np.random.normal(0, scale/len(self.edge_innov))

    def mutate(self, scale, edge_mute_prob=0.1, node_mute_prob=0.5):
        #mutate the weights
        self.mutate_weights(scale)

        #mutate the topology of the net
        if np.random.random()<edge_mute_prob:
            not_added=True
            while not_added:
                chi = np.random.randint(0, len(self.nodes))
                par = np.random.randint(0, len(self.nodes))
                not_added = not self.add_edge(par, chi, 0, np.random.normal(0, scale))
        if np.random.random()<node_mute_prob:
            chi = np.random.randint(0, len(self.nodes))
            par = np.random.randint(0, len(self.nodes))
            self.add_node(par, chi, activation=sigmoid)






def test_neat():
    # A test function to ensure that the stuff in the Neat module works as intended

    neat = Neat(2, 2, identity)
    # print(neat.activate([1, 2]))
    try:
        assert not neat.find_path(0, 1)
        assert not neat.find_path(2, 3)
        assert not neat.find_path(0, 3)

        neat.add_edge(0, 3, 0, val=np.random.normal(0, 0.2))
        neat.add_edge(1, 3, 0, val=np.random.normal(0, 0.2))
        neat.add_edge(1, 4, 0, val=np.random.normal(0, 0.2))
        print(neat.edge_adj)
        neat.add_node(1, 4)
        print(neat.edge_adj)
        print(neat.edge_innov)
        print(neat.edge_adj)
        # assert neat.find_path(0, 3)
    except AssertionError:
        print("Error in pathfinder")
    # print(neat.activate([1, 1]))
    print(neat.activate([1, 0]))

    print(neat.activate([1, 0]))


def timetest():
    neat = Neat(2, 2, tanh)
    for i in range(100):
        neat.mutate(0.2)
        print(neat.edge_innov)


if __name__ == "__main__":
    timetest()
    #cProfile.run("timetest()")
