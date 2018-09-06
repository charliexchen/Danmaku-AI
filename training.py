import numpy as np
import copy
from objects import environ
from display import gui
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
    #This object tracks the timing
    def __init__(self):
        self.reset()
    def reset(self):
        self.start_time = time.time()
    def elapsed(self, l_time):
        output = ""
        if l_time >= 3600:
            hours = int(math.floor(l_time / (60 * 60)))
            l_time -= hours*60*60
            output += "{} hr ".format(hours)
        if l_time >=60:
            minutes = int(math.floor(l_time / (60)))
            l_time -= minutes*60
            output += "{} min ".format(minutes)
        return output+ "{0:.2f} sec ".format(l_time)
    def elapsed_time(self, text="Time Elapsed"):
        print(text+ ": {}".format(self.elapsed(time.time() - self.start_time)))

class population():
    def __init__(self, sensorpos, pop_size=100, multithread=False, procs=cpu_count()):
        #initialise the population of agents
        self.agents = []
        for i in range(pop_size):


            self.agents.append(environ((sensorpos, Neat(25, 2, tanh)), bullet_types={"aimed": 15, "spiral": 1, "random": 1}))


            '''    
            self.agents[-1].controller.add_layer(14, relu)
            self.agents[-1].controller.add_layer(8, sigmoid)
            self.agents[-1].controller.add_layer(2, tanh)
            '''
        self.pop_size = pop_size
        self.gen=1
        self.training_timer = Timer()
        #associate a pool of workers with the population
        print("Training with {} processes".format(procs))
        self.pool = Pool(processes=procs)

    def find_fitness(self, max_time):
        self.training_timer.reset()

        #wrapper for function in order to make use of multiprocessing

        #evaluate fitness of each agent
        self.fitness=self.pool.map(f, self.agents)

        #assign the fitness value to each agent
        for i in range(self.pop_size):
            self.agents[i].fitness = self.fitness[i]

        #print time taken to evaluate fitness
        self.training_timer.elapsed_time("Fitness Evaluation Time")


    def select(self):
        list.sort(self.agents, key=lambda x: x.fitness)
        print("Lowest fitness: {}".format(self.agents[0].fitness))
        print("Highest fitness: {}".format(self.agents[-1].fitness))
        print("Top 10 agents fitness: {}".format([agent.fitness for agent in self.agents][-10:]))
        new_agents = []
        for i in range(int(self.pop_size * (9 / 10)), self.pop_size):
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
        filepath=os.join(destination, filename)
        print("Saving generation {} to {}".format(self.gen, filepath ))
        pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
        print("Saving generation {}".format(self.gen))

    def load_generation(self, filename):
        print("saving gen {}".format(i))
        pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
        print("saved gen {}".format(i))





def train(params, import_generation=True):
    pass

if __name__=="__main__":

    sensor_pos = [
        (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
        (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
        (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
        (0, -40, 1), (10, -20, 1), (-10, -20, 1), (-40, 0, 1), (40, 0, 1),
        (0, 15, 1), (10, -30, 1), (-10, -30, 1)]

    pop = population(sensor_pos, 100)

    starting_gen=-1
    if starting_gen>-1:
        trained_pop = pickle.load(open("generation{}.p".format(starting_gen), "rb"))
        pop.agents = trained_pop

    rate = [0.25, 0.2]

    GUI = gui()
    for i in range(starting_gen+1,1000):
        print("Evaluating fitness...")
        pop.find_fitness(1000)
        print("Selecting fittest")
        pop.select()
        print("Surviving agents: {}".format(len(pop.agents)))
        pop.breed([rate[0] / (i + 1), rate[1] / (i + 1)])
        print("saving gen {}".format(i))
        pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
        print("saved gen {}".format(i))
        if i%5==0:
            GUI.display_imported_generation("generation{}.p".format(i), 3)
        # if i % 25 == 0 and i != 0:
        #    rate = [i / 2 for i in rate]
        #    print("Decaying learning rate to: {}".format(rate))
