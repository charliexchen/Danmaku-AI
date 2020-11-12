import numpy as np
import gym
import copy
from Learning.neural_net import dense_net, relu, tanh, sigmoid
import time
import math, pickle


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


class population:
    def __init__(self, pop_size=200, max_reward=1000):
        # initialise the population of agents
        self.env = gym.make("LunarLander-v2")

        self.max_reward = max_reward
        agent = dense_net(8, 10, relu, recursive=True, rec_size=8)
        agent.add_layer(10, relu)
        agent.add_layer(4, sigmoid, initvar=0.0001, final=True)
        self.agents = []
        for i in range(pop_size):
            new_agent = copy.deepcopy(agent)
            if 1 > 0:
                new_agent.mutate(0.1)
            self.agents.append({"controller": new_agent, "fitness": 0})
        self.pop_size = pop_size
        self.gen = 1
        self.training_timer = Timer()

    def find_fitness(self):
        self.training_timer.reset()
        self.fitness = []
        for index, agent in enumerate(self.agents):
            total_reward = 0
            observation = self.env.reset()
            for i in range(5):
                for time in range(self.max_reward):
                    action = np.argmax(agent["controller"].activate(observation))
                    observation, reward, done, info = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
            agent["fitness"] = total_reward
        # print time taken to evaluate fitness
        self.training_timer.elapsed_time("Fitness Evaluation Time")

    def select(self):
        print(self.agents)
        list.sort(self.agents, key=lambda x: x["fitness"])
        print("Lowest fitness: {}".format(self.agents[0]["fitness"]))
        print("Highest fitness: {}".format(self.agents[-1]["fitness"]))
        print(
            "Top 10 agents fitness: {}".format(
                [agent["fitness"] for agent in self.agents][-10:]
            )
        )
        print(
            "Average fitness: {}".format(
                np.mean([agent["fitness"] for agent in self.agents])
            )
        )
        print(
            "Variance of fitness: {}".format(
                np.var([agent["fitness"] for agent in self.agents])
            )
        )
        self.agents = self.agents[int(self.pop_size * (99 / 100)) :]
        return self.agents[-1]

    def breed(self, rate):
        i = 0
        self.agents = self.agents[::-1]
        while len(self.agents) < self.pop_size:
            self.agents.append(copy.deepcopy(self.agents[i]))
            self.agents[-1]["controller"].mutate(rate)
            i += 1
        self.training_timer.elapsed_time("Total Generation Time")


env = gym.make("LunarLander-v2")
env.reset()
if __name__ == "__main__":
    save_path = "cartpole_nets"

    pop = population(1000)

    rate = 0.2

    """
    for i in range(10000):
        print(f"Generation {i}")
        print("Evaluating fitness...")
        pop.find_fitness()
        print("Selecting fittest")
        best = pop.select()
        print("Surviving agents: {}".format(len(pop.agents)))
        pop.breed(rate/ ((1 + i)))

        done = False
        observation = env.reset()
        if 1 == i%100:
            pickle.dump(best, open(f"lunar_generation{i}.p", "wb"))
    """
    best = pickle.load(open("lunar_generation601.p", "rb"))
    observation = env.reset()
    while True:
        env.render()
        action = np.argmax(best["controller"].activate(observation))
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
