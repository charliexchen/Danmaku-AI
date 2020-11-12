import math


def is_square(x):
    return int(math.sqrt(x)) ** 2 == x


for i in range(-1, 10000000):
    if is_square(i ** 3 + 2):
        print(i)

print("end")
