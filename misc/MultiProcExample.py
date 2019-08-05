from multiprocessing import Pool, TimeoutError, cpu_count
import time
from Game.objects import environ
from Learning.neural_net import dense_net, relu, tanh


class timer_cons:
    def __init__(self):
        self.fitness = 0

    def eval_fitness(self):
        time.sleep(1)
        self.fitness = 1
        return 5


core_count = cpu_count()


def f(env):
    return env.eval_fitness(500)


# print (list(map(f, agents)))


def func(**kwargs):
    for key, value in kwargs.items():
        print(key, value)


if __name__ == "__main__":
    params = {"a": 1, "b": 2}
    func(a=1, b=2)
    func()
    for procs in range(1, 10):

        nets = [dense_net(16, 10, relu, recursive=True) for i in range(500)]
        for net in nets:
            net.add_layer(2, tanh)
        agents = [environ((8, net)) for net in nets]

        print(nets[0].layers[0].weights[0])
        with Pool(processes=procs) as pool:
            t = time.time()
            print(pool.map(f, agents))
            t = time.time() - t
            print("Time taken for {} processes: {}".format(procs, t))
            print("Effeciency: {}".format(t / procs))
        print(nets)
        print(nets[0].layers[0].weights[0])
        # print([agent.fitness for agent in agents])

    # exiting the 'with'-block has stopped the pool
