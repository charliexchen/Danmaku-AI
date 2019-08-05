import numpy as np
import copy
from Game.objects import environ
from Learning.neural_net import dense_net, relu, tanh
import pickle
from multiprocessing import Pool, cpu_count
import os
import time
import math
from Game.enums import sensorType, bulletType


class Timer:
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


class Logger:
    pass


class population:
    def __init__(
        self, sensors, pop_size=100, agent=None, multithread=False, procs=cpu_count()
    ):
        # initialise the population of agents

        self.agents = []

        if agent == None:
            input_len = 0
            if sensorType.POINT in sensors:
                input_len += len(sensors[sensorType.POINT])
            if sensorType.PROXIMITY in sensors:
                input_len += 2 * sensors[sensorType.PROXIMITY]
            if sensorType.LOCATION in sensors:
                input_len += 2
            if sensorType.LINE in sensors:
                input_len += sensors[sensorType.LINE]
            agent = environ(
                (sensors, dense_net(input_len, 64, relu, recursive=True, rec_size=16)),
                bullet_types={
                    bulletType.AIMED: 15,
                    bulletType.SPIRAL: 1,
                    bulletType.RANDOM: 1,
                },
            )
            agent.controller.add_layer(64, relu)
            agent.controller.add_layer(32, relu)
            agent.controller.add_layer(16, relu)
            agent.controller.add_layer(64, relu)
            agent.controller.add_layer(2, tanh, final=True)

        for i in range(pop_size):
            self.agents.append(copy.deepcopy(agent))
        self.pop_size = pop_size
        self.gen = 1
        self.training_timer = Timer()
        # associate a pool of workers with the population
        print("Training with {} processes".format(procs))
        self.pool = Pool(processes=procs)

    @staticmethod
    def find_fitness(env):
        # Wrapper for fitness function in order to make use of multiprocessing
        return env.eval_dmg(1200)

    def find_population_fitness(self):
        self.training_timer.reset()

        # Evaluate fitness of each agent, with multiprocessing
        self.fitness = self.pool.map(population.find_fitness, self.agents)

        # Assign the fitness value to each agent
        for i in range(self.pop_size):
            self.agents[i].fitness = self.fitness[i]

        # Print time taken to evaluate fitness
        self.training_timer.elapsed_time("Fitness Evaluation Time")

    def select(self):
        # Order agents by fitness
        list.sort(self.agents, key=lambda x: x.fitness)

        # Prinst some useful info
        print("Lowest fitness: {}".format(self.agents[0].fitness))
        print("Highest fitness: {}".format(self.agents[-1].fitness))
        print(
            "Top 10 agents fitness: {}".format(
                [agent.fitness for agent in self.agents][-10:]
            )
        )
        print(
            "Average fitness: {}".format(
                np.mean([agent.fitness for agent in self.agents])
            )
        )
        print(
            "Variance of fitness: {}".format(
                np.var([agent.fitness for agent in self.agents])
            )
        )

        # Cull the weaklings
        new_agents = []
        print(self.pop_size * (99 / 100))
        for i in range(int(self.pop_size * (90 / 100)), self.pop_size):
            if (i + 1) / self.pop_size > np.random.uniform(0, 1.0):
                new_agents.append(self.agents[i])
        self.agents = new_agents

    def breed(self, rate):
        # add mutated copies of what's left until we have a full population again
        i = 0
        self.agents = self.agents[::-1]
        while len(self.agents) < self.pop_size:
            self.agents.append(copy.deepcopy(self.agents[i]))
            self.agents[-1].controller.mutate(rate[0])
            self.agents[-1].fighter.mutate_sensors(rate[1])
            i += 1
        self.training_timer.elapsed_time("Total Generation Time")

    def save_generation(self, directory, filename):
        filepath = os.path.join(directory, filename)
        print(filepath)
        print("Saving generation {} to {}".format(self.gen, filepath))
        fitness = [env.fitness for env in self.agents]
        nets = [env.controller for env in self.agents]
        bullet_type = self.agents[0].bullet_cooldowns
        sensor_type = self.agents[0].sensors
        output_dict = {
            "fitness": fitness,
            "nets": nets,
            "bullet_type": bullet_type,
            "sensor_type": sensor_type,
        }
        pickle.dump(output_dict, open(filepath, "wb"))
        print(output_dict)
        print("Saving generation {}".format(self.gen))

    def load_generation(self, filename):
        if i % 5 == 0:
            print("saving gen {}".format(i))
            pickle.dump(pop.agents, open("generation{}.p".format(i), "wb"))
            print("saved gen {}".format(i))


if __name__ == "__main__":
    save_path = "saved_nets"
    sensor_pos = {
        sensorType.POINT: [
            (0, -10, 1),
            (10, -10, 1),
            (-10, -10, 1),
            (-10, 0, 1),
            (10, 0, 1),
            (0, -20, 1),
            (20, -20, 1),
            (-20, -20, 1),
            (-20, 0, 1),
            (20, 0, 1),
            (0, -30, 1),
            (30, -30, 1),
            (-30, -30, 1),
            (-30, 0, 1),
            (30, 0, 1),
            (10, -20, 1),
            (-10, -20, 1),
            (0, 15, 1),
            (10, -30, 1),
            (-10, -30, 1),
            (0, -3, 1),
            (3, 0, 1),
            (-3, 0, 1),
        ],
        sensorType.PROXIMITY: 3,
        sensorType.LOCATION: True,
        sensorType.LINE: 16,
    }

    pop = population(sensor_pos, 100)

    starting_gen = input(
        "Start from which existing generation? (return empty if no such generation exists):"
    )
    if starting_gen == "":
        starting_gen = -1
    else:
        try:
            starting_gen = int(starting_gen)
        except:
            print("Error -- input not an integer")

    if starting_gen > -1:
        file_name = os.path.join(save_path, "generation{}.p".format(starting_gen))
        trained_nets = pickle.load(open(file_name, "rb"))
        for i in range(len(pop.agents)):
            pop.agents[i].controller = trained_nets["nets"][i]

    rate = [0.25, 0.2]

    display = input(
        "Display best in generation? (return empty if no, else return anything)"
    )
    display = display != ""
    if display:
        from Game.display import gui

        GUI = gui()

    for i in range(starting_gen + 1, 10000):
        print("Evaluating fitness...")
        pop.find_population_fitness()
        print("Selecting fittest")
        pop.select()
        print("Surviving agents: {}".format(len(pop.agents)))
        pop.breed([rate[0] / ((1 + i)), rate[1] / ((1 + i))])
        if i % 5 == 0:
            file_name = os.path.join(save_path, "generation{}.p".format(i))
            pop.save_generation(save_path, "generation{}.p".format(i))
            if display:
                GUI.display_imported_generation(file_name, 3)
        # if i % 25 == 0 and i != 0:
        #    rate = [i / 2 for i in rate]
        #    print("Decaying learning rate to: {}".format(rate))
