import math


def sqdist(p1, p2):
    disvec = [p1[i] - p2[i] for i in range(2)]
    return sum([i ** 2 for i in disvec])


def dist(p1, p2):
    return math.sqrt(sqdist(p1, p2))


def dot(v1, v2):
    assert len(v1) == len(
        v2
    ), "Error -- dot product has different lengths {} and {}".format(len(v1), len(v2))
    return sum([v1[i] + v2[i] for i in range(len(v1))])


def angle(p1, p2):
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    return math.atan2(dy, dx)


def col(p1, p2, r1, r2):
    return sqdist(p1, p2) < (r1 + r2) ** 2


def sub_angle(a1, a2):
    output = a1 - a2
    if output > math.pi:
        output -= 2 * math.pi
    if output < -math.pi:
        output += 2 * math.pi
    return output


def outofbound(pos, boundary, rad=0):
    if (
        pos[0] < -rad
        or pos[0] > boundary[0] + rad
        or pos[1] < -rad
        or pos[1] > boundary[1] + rad
    ):
        return True
    return False



