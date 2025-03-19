import numpy as np

def LCG(a,c,m, seed, n):
    x_0 = seed
    random_numbers = []

    for _ in range(n):
        x_0 = (a*x_0 + c) % m
        random_numbers.append(x_0/m)

    return random_numbers

