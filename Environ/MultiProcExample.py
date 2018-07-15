from multiprocessing import Pool, TimeoutError, cpu_count
import time
from Environ.objects import environ
from Environ.neural_net import dense_net, relu, sigmoid, tanh


class timer_cons:
    def __init__(self):
        self.fitness=0
    def eval_fitness(self):
        time.sleep(1)
        self.fitness=1
        return 5


core_count =cpu_count()


def f(env):
    return env.eval_fitness(100)
#print (list(map(f, agents)))

if __name__ == '__main__':
    # start 4 worker processes
    for procs in range(1,10):
        nets = [dense_net(16, 10, relu, recursive=True) for i in range(500)]
        for net in nets:
            net.add_layer(2, tanh)
        agents = [environ((8, net)) for net in nets]


        with Pool(processes=procs) as pool:
            t=time.time()
            print(pool.map(f, agents))
            t = time.time()-t
            print ('Time taken for {} processes: {}'.format(procs, t))
            print ('Effecienty: {}'.format(t/procs))
        #print([agent.fitness for agent in agents])

    # exiting the 'with'-block has stopped the pool