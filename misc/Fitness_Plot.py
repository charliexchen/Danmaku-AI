import pickle
from Learning import *
import numpy as np

fitness_data = pickle.load(open("fitness_data.p", "rb"))
data = []
x = []
y = []
for key in fitness_data.keys():
    letters = "qwertyuiopasdfghjklzxcvbnm."
    number = ""
    for i in str(key):
        if i not in letters:
            number += i
    number = int(number)
    x.append(number)
    y.append(np.mean(fitness_data[key]))
    data.append([number, np.mean(fitness_data[key])])

list.sort(data, key=lambda i: i[0])
X = [i[0] for i in data]
Y = [i[1] for i in data]
import matplotlib.pyplot as plt
import numpy as np

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
for n in [1, 2, 3, 4]:
    plt.plot(X, Y)

plt.show()
list.sort(y, key=lambda i: x[y.index(i)])
