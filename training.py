import numpy as np
import copy
from objects import environ
from neural_net import dense_net, relu, sigmoid, tanh
from Neat import Neat
import pickle
from multiprocessing import Pool, TimeoutError, cpu_count
import os
import time
import math


def f(env):
    return env.eval_fitness(1000)

class Timer():
    # This object tracks the timing
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self, l_time):
        output = ""
        if l_time >= 3600:
            hours = int(math.floor(l_time / (60 * 60)))
            l_time -= hours * 60 * 60
            output += "{} hr ".format(hours)
        if l_time >= 60:
            minutes = int(math.floor(l_time / (60)))
            l_time -= minutes * 60
            output += "{} min ".format(minutes)
        return output + "{0:.2f} sec ".format(l_time)

    def elapsed_time(self, text="Time Elapsed"):
        print(text + ": {}".format(self.elapsed(time.time() - self.start_time)))

class population():
    def __init__(self, sensors, pop_size=100, multithread=False, procs=cpu_count()):
        # initialise the population of agents
        self.agents = []
        input_len = 0
        if "point" in sensors:
            input_len += len(sensors["point"])
        if "prox" in sensors:
            input_len += 2 * sensors["prox"]
        if "loc" in sensors:
            input_len += 2
        if "line" in sensors:
            input_len += sensors["line"]
        for i in range(pop_size):
            self.agents.append(environ((sensors, dense_net(input_len, 64, relu, recursive=True, rec_size=16)),
                                       bullet_types={"aimed": 15, "spiral": 1, "random": 1}))
            # Neat(25, 2, tanh)
            self.agents[-1].controller.add_layer(64, relu)
            self.agents[-1].controller.add_layer(32, relu)

            self.agents[-1].controller.add_layer(16, relu)
            self.agents[-1].controller.add_layer(2, tanh, final=True)

        self.pop_size = pop_size
        self.gen = 1
        self.training_timer = Timer()
        # associate a pool of workers with the population
        print("Training with {} processes".format(procs))
        self.pool = Pool(processes=procs)

    def find_fitness(self, max_time):
        self.training_timer.reset()

        # wrapper for function in order to make use of multiprocessing

        # evaluate fitness of each agent
        self.fitness = self.pool.map(f, self.agents)

        # assign the fitness value to each agent
        for i in range(self.pop_size):
            self.agents[i].fitness = self.fitness[i]

        # print time taken to evaluate fitness
        self.training_timer.elapsed_time("Fitness Evaluation Time")

    def select(self):
        list.sort(self.agents, key=lambda x: x.fitness)
        print("Lowest fitness: {}".format(self.agents[0].fitness))
        print("Highest fitness: {}".format(self.agents[-1].fitness))
        print("Top 10 agents fitness: {}".format([agent.fitness for agent in self.agents][-10:]))
        print("Average fitness: {}".format(np.mean([agent.fitness for agent in self.agents])))
        print("Variance of fitness: {}".format(np.var([agent.fitness for agent in self.agents])))
        new_agents = []
        print(self.pop_size * (99 / 100))
        for i in range(int(self.pop_size * (99 / 100)), self.pop_size):
            if (i + 1) / self.pop_size > np.random.uniform(0, 1.0):
                new_agents.append(self.agents[i])
        self.agents = new_agents

    def breed(self, rate):
        i = 0
        self.agents = self.agents[::-1]
        while len(self.agents) < self.pop_size:
            self.agents.append(copy.deepcopy(self.agents[i]))
            self.agents[-1].controller.mutate(rate[0])
            self.agents[-1].fighter.mutate_sensors(rate[1])
            i += 1

        self.training_timer.elapsed_time("Total Generation Time")

    def save_generation(self, filename, destination):
        filepath = os.join(destination, filename)
        print("Saving generation {} to {}".format(self.gen, filepath))
        pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
        print("Saving generation {}".format(self.gen))

    def load_generation(self, filename):
        print("saving gen {}".format(i))
        pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
        print("saved gen {}".format(i))


if __name__ == "__main__":

    save_path = "saved_nets"

    sensor_pos = {"point": [
        (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
        (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
        (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
        (10, -20, 1), (-10, -20, 1),
        (0, 15, 1), (10, -30, 1), (-10, -30, 1), (0, -3, 1), (3, 0, 1), (-3, 0, 1)],
        "prox": 3, "loc": True, "line": 16}

    #sensor_pos = {"loc":True, "line": 16}
    pop = population(sensor_pos, 1000)

    starting_gen = input("Start from which existing generation? (return empty if no such generation exists):")
    if starting_gen == "":
        starting_gen = -1
    else:
        try:
            starting_gen = int(starting_gen)
        except:
            print("Error -- input not an integer")

    if starting_gen > -1:
        file_name = os.path.join(save_path, "generation{}.p".format(starting_gen))
        trained_pop = pickle.load(open(file_name, "rb"))
        pop.agents = trained_pop

    rate = [0.25, 0.2]

    display = input("Display best in generation? (return empty if no)")
    display = display != ""
    if display:
        from display import gui

        GUI = gui()

    for i in range(starting_gen + 1, 10000):
        print("Evaluating fitness...")
        pop.find_fitness(1000)
        print("Selecting fittest")
        pop.select()
        print("Surviving agents: {}".format(len(pop.agents)))
        pop.breed([rate[0] / ((1 + i)), rate[1] / ((1 + i))])
        print("saving gen {}".format(i))

        file_name = os.path.join(save_path, "generation{}.p".format(i))

        pickle.dump(pop.agents, open(file_name, "wb"))
        print("saved gen {}".format(i))
        if i % 5 == 0 and display:
            GUI.display_imported_generation(file_name, 3)
        # if i % 25 == 0 and i != 0:
        #    rate = [i / 2 for i in rate]
        #    print("Decaying learning rate to: {}".format(rate))
