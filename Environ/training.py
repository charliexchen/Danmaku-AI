import numpy as np
import copy
from Environ.objects import environ
from Environ.neural_net import dense_net, relu, sigmoid, tanh
import pickle
from multiprocessing import Pool, TimeoutError, cpu_count
import os
from multiprocessing import Pool, TimeoutError


class population():
    def __init__(self, sensorpos, pop_size=100, multithread=False):
        self.agents = []
        for i in range(pop_size):
            self.agents.append(environ((sensorpos, dense_net(16, 10, relu)), cd=15))
            self.agents[-1].controller.add_layer(10, tanh)
            self.agents[-1].controller.add_layer(5, tanh)
            self.agents[-1].controller.add_layer(2, tanh)
        self.pop_size = pop_size

    def find_fitness(self, max_time, processes=cpu_count()):
        for agent in self.agents:
            agent.eval_fitness(max_time)

    def select(self):
        list.sort(self.agents, key=lambda x: x.fitness)
        print("Lowest fitness: {}".format(self.agents[0].fitness))
        print("Highest fitness: {}".format(self.agents[-1].fitness))
        print("Top 10 agents fitness: {}".format([agent.fitness for agent in self.agents][-10:]))
        new_agents = []
        for i in range(int(self.pop_size * (7 / 10)), self.pop_size):
            if (i + 1) / self.pop_size > np.random.uniform(0, 1.0):
                new_agents.append(self.agents[i])
        self.agents = new_agents

    def breed(self, rate):
        i = 0
        self.agents = self.agents[::-1]
        while len(self.agents) < self.pop_size:
            self.agents.append(copy.deepcopy(self.agents[i]))
            self.agents[-1].controller.mutate(rate[0])
            # self.agents[-1].fighter.mutate_sensors(rate[1])
            i += 1


sensorpos1 = [
    (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
    (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
    (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
    (0, -40, 1), (10, -20, 1), (-10, -20, 1), (-40, 0, 1), (40, 0, 1),
    (0, 15, 1), (10, -30, 1), (-10, -30, 1)]

# trained_pop = pickle.load(open("generation239.p", "rb"))
# pop.agents = trained_pop

def train(params, import_generation=True):
    pass

pop = population(8, 10)

rate = [0.5, 0.5]

baserate = 0.2
for i in range(1000):
    print("Evaluating fitness...")
    pop.find_fitness(1000)
    print("Selecting fittest")
    pop.select()
    print("Surviving agents: {}".format(len(pop.agents)))
    pop.breed([rate[0] / (i + 1), rate[1] / (i + 1)])
    print("saving gen {}".format(i))
    pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
    print("saved gen {}".format(i))
    # if i % 25 == 0 and i != 0:
    #    rate = [i / 2 for i in rate]
    #    print("Decaying learning rate to: {}".format(rate))
