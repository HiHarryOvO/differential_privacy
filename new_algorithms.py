"""
Implement PM and HM mechanisms and apply them to empirical risk minimization.
"""
from scipy.special import comb
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from math import exp
import itertools


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


# tp is a one-dimensional tuple
def duchi_single_attr_method(tp, epsilon):
    t_star = (exp(epsilon) + 1) / (exp(epsilon) - 1)
    prob = (exp(epsilon) - 1) * tp / (2 * exp(epsilon) + 2) + 1 / 2
    if generate_binary_random(prob, 1, 0) == 1:
        return t_star
    else:
        return (-1) * t_star


def piecewise_mechanism(tp, epsilon):  # tp is a one-dimensional tuple
    C = (exp(epsilon / 2) + 1) / (exp(epsilon / 2) - 1)
    l_ti = (C + 1) * tp / 2 - (C - 1) / 2
    r_ti = l_ti + C - 1

    if random.uniform(0, 1) < exp(epsilon / 2) / (exp(epsilon / 2) + 1):
        return random.uniform(l_ti, r_ti)
    else:
        sample = random.uniform(0, C + 1)
        if sample <= (l_ti + C):
            return sample - C
        else:
            return sample - 1


def hybrid_mechanism(tp, epsilon):  # tp is a one-dimensional tuple
    if epsilon > 0.61:
        alpha = 1 - exp(-1 * epsilon / 2)
    else:
        alpha = 0
    if random.random() < alpha:
        return piecewise_mechanism(tp, epsilon)
    else:
        return duchi_single_attr_method(tp, epsilon)


def duchi_method(tp, epsilon):  # tp is a d-dimensional tuple
    d = len(tp)
    if d % 2 != 0:
        C_d = 2 ** (d - 1) / comb(d - 1, int((d - 1) / 2))
    else:
        C_d = (2 ** (d - 1) + comb(d, int(d / 2))) / comb(d - 1, int(d / 2))

    B = C_d * (exp(epsilon) + 1) / (exp(epsilon) - 1)
    neg_B = (-1) * B
    v = [generate_binary_random(0.5 + 0.5 * tp[j], 1, -1) for j in range(d)]

    t_pos = []
    t_neg = []
    for t_star in itertools.product([neg_B, B], repeat=d):
        if np.dot(np.array(t_star), np.array(v)) > 0:
            t_pos.append(t_star)
        else:
            t_neg.append(t_star)

    if generate_binary_random(math.exp(epsilon) / (math.exp(epsilon) + 1), 1, 0) == 1:
        return random.choice(t_pos)
    else:
        return random.choice(t_neg)


def proposed_method(tp, epsilon, method):  # tp is a d-dimensional tuple
    d = len(tp)
    k = max(1, min(d, int(epsilon / 2.5)))
    samples = random.sample(list(range(0, d)), k)
    t_star = tp[:]
    for j in samples:
        if method == "pm":
            t_star[j] = (d / k) * piecewise_mechanism(tp[j], epsilon)
        elif method == "hm":
            t_star[j] = (d / k) * hybrid_mechanism(tp[j], epsilon)
        else:
            raise TypeError("method should either be 'pm' or 'hm'")
    return t_star


if __name__ == '__main__':
    random.seed(10)
    num = 5000
    t1 = [random.random() for i in range(num)]
    mean_t1 = np.mean(t1)
    x = []
    relative_error_duchi = []
    relative_error_pm = []
    relative_error_hm = []
    for epsilon_10 in range(10, 101, 5):
        epsilon = epsilon_10 / 10
        x.append(epsilon)

        private_t_duchi = [duchi_single_attr_method(tp, epsilon) for tp in t1]
        private_t_pm = [piecewise_mechanism(tp, epsilon) for tp in t1]
        private_t_hm = [hybrid_mechanism(tp, epsilon) for tp in t1]

        relative_error_duchi.append(math.fabs(mean_t1 - np.mean(private_t_duchi)))
        relative_error_pm.append(math.fabs(mean_t1 - np.mean(private_t_pm)))
        relative_error_hm.append(math.fabs(mean_t1 - np.mean(private_t_hm)))

    plt.plot(x, relative_error_duchi, color='red', label='duchi')
    plt.plot(x, relative_error_pm, color='blue', label='pm')
    plt.plot(x, relative_error_hm, color='green', label='hm')
    plt.xlabel("epsilon")
    plt.ylabel("relative error")
    plt.legend()
    plt.show()
