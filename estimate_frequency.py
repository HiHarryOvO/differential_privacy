"""
This file is based on part 3.2 of Collecting and Analyzing Data from Smart Device Users with Local Differential Privacy
written by the group from NTU and HBKU.

Estimating frequencies for categorical attributes.
"""
import math
import random
import numpy as np


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


# k refers to the number of possible values for this attribute and n refers to the number of users
# tp refers to a column of the the dataset representing a attribute
def bassily_method(tp, k, epsilon, beta):
    n = len(tp)
    frequency_estimate = []
    gamma = math.sqrt(math.log(2 * k / beta) / (epsilon ** 2 * n))
    m = round(math.log(k + 1) * math.log(2 / beta) / (gamma ** 2))  # it might be possible that m = 0?
    print("m =", m)
    matrix_value = [-1 / math.sqrt(m), 1 / math.sqrt(m)]
    phi = [[random.choice(matrix_value) for col in range(k)] for row in range(m)]
    z_sum = [0 for i_1 in range(m)]
    for i in range(n):
        s = random.randint(0, m - 1)
        c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        if generate_binary_random(math.exp(epsilon) / (math.exp(epsilon) + 1), 1, 0) == 1:
            alpha = c * m * phi[s][tp[i] - 1]
        else:
            alpha = -1 * c * m * phi[s][tp[i] - 1]
        z_sum[s] += alpha

    z_mean = np.array(z_sum) / n
    for l in range(k):
        element = np.dot(np.array(phi)[:, l], z_mean)
        if element >= 0:
            frequency_estimate.append(element)
        else:
            frequency_estimate.append(0)
    print("sum =", sum(frequency_estimate))
    return frequency_estimate


tp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
print(bassily_method(tp, 100, 10, 0.05))
print("error bound:", math.sqrt(math.log(100/0.05)) / (10 * math.sqrt(13)))
