"""
This file is based on part 3.1 of Collecting and Analyzing Data from Smart Device Users with Local Differential Privacy
written by the group from NTU and HBKU.

Estimating mean values for numeric attributes.

We implement Duchi et al's method and Proposed method put forward by the authors.
"""
from scipy.special import comb
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import itertools


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


# tp refers to a row of the dataset representing the data of a user
def duchi_method(tp, epsilon):
    d = len(tp)
    if d % 2 != 0:
        C_d = 2 ** (d - 1)
        B = (2 ** d + C_d * (math.exp(epsilon) - 1)) / (comb(d - 1, int((d - 1) / 2)) * (math.exp(epsilon) - 1))
    else:
        C_d = 2 ** (d - 1) - comb(d, int(d / 2))
        B = (2 ** d + C_d * (math.exp(epsilon) - 1)) / (comb(d - 1, int(d / 2)) * (math.exp(epsilon) - 1))

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


def proposed_method(tp, epsilon):
    d = len(tp)
    tp_star = [0 for i in range(d)]
    j = random.randint(0, d - 1)
    pr = (tp[j] * (math.exp(epsilon) - 1) + math.exp(epsilon) + 1) / (2 * math.exp(epsilon) + 2)
    value = d * (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    if generate_binary_random(pr, 1, 0) == 1:
        tp_star[j] = value
    else:
        tp_star[j] = -1 * value
    return tp_star


random.seed(10)
dimension = 10
num = 5000
epsilon = 0.7

sample_t = np.array([[random.uniform(-1, 1) for di in range(dimension)] for n in range(num)])
sample_mean = np.mean(sample_t, axis=0)
# print("the real sample mean is", sample_mean, "\n")

duchi_method_t = np.array([duchi_method(tp, epsilon) for tp in sample_t])
proposed_method_t = np.array([proposed_method(tp, epsilon) for tp in sample_t])

duchi_method_mean = np.mean(duchi_method_t, axis=0)
proposed_method_mean = np.mean(proposed_method_t, axis=0)

# print("mean using Duchi's method:", duchi_method_mean)
# print("mean using proposed method:", proposed_method_mean, "\n")

# print("absolute error of Duchi's method:", np.fabs(sample_mean - duchi_method_mean))
# print("absolute error of proposed method:", np.fabs(sample_mean - proposed_method_mean), "\n")

# print("relative error of Duchi's method:", np.fabs(np.true_divide(sample_mean - duchi_method_mean, sample_mean)))
# print("relative error of proposed method:", np.fabs(np.true_divide(sample_mean - proposed_method_mean, sample_mean)))

plt.plot(np.fabs(sample_mean - duchi_method_mean), marker="o", label="duchi")
plt.plot(np.fabs(sample_mean - proposed_method_mean), marker="o", color="red", label="proposed")
plt.title("Absolute Error")
plt.legend()
plt.savefig("absolute_error_1.png")
plt.figure()

plt.plot(np.fabs(np.true_divide(sample_mean - duchi_method_mean, sample_mean)), marker="o", label="duchi")
plt.plot(np.fabs(np.true_divide(sample_mean - proposed_method_mean, sample_mean)), marker="o",
         color="red", label="proposed")
plt.legend()
plt.title("Relative Error")
plt.savefig("relative_error_1.png")
plt.figure()

y = []
for i in range(1, 11):
    epsilon = i / 10
    sample_t = np.array([[random.uniform(-1, 1) for di in range(dimension)] for n in range(num)])
    sample_mean = np.mean(sample_t, axis=0)
    # print("the real sample mean is", sample_mean, "\n")

    proposed_method_t = np.array([proposed_method(tp, epsilon) for tp in sample_t])

    proposed_method_mean = np.mean(proposed_method_t, axis=0)
    y.append(np.mean(np.fabs(np.true_divide(sample_mean - proposed_method_mean, sample_mean))))

plt.plot(y, marker="o")
plt.title("Relationship between epsilon and RE")
plt.xlabel("epsilon")
plt.ylabel("Relative Error")
# plt.savefig("epsilon.png")
plt.show()
