from multiprocessing import Pool
import time
def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))

def do_something(t, taskname):
    print("starting task {}".format(taskname))
    print 
