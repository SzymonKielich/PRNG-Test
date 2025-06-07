import time as time_module
import numpy as np
import math
import random
import scipy.stats as stats
from typing import Type, Union
# import struct

from numpy.random import Generator, PCG64, MT19937, Philox, SFC64


def numpy_random(n, algorithm: Type[Union[PCG64, MT19937, Philox, SFC64]] = PCG64):
    rng = Generator(algorithm())
    return rng.uniform(0, 1, n)


def LCG(a, c, m, n, seed=None):
    if seed is None:
        seed = int(time_module.time() * 1000000) % m
    x_0 = seed
    random_numbers = []

    for _ in range(n):
        x_0 = (a * x_0 + c) % m
        random_numbers.append(x_0 / m)

    return random_numbers


def python_random(n):
    return [random.random() for _ in range(n)]


def system_random(n):
    sys_random = random.SystemRandom()
    return [sys_random.random() for _ in range(n)]


def xorshift(n, seed=None):
    state = seed
    if seed is None:
        state = int(time_module.time() * 1000000) & 0xFFFFFFFF

    mask = 0xFFFFFFFF
    results = []
    for _ in range(n):
        state ^= (state << 13) & mask
        state ^= (state >> 17) & mask
        state ^= (state << 5) & mask
        results.append(state / (mask + 1))
    return results

# def save_as_binary(data, filename="rng_output.bin"):
#     """Save random numbers as raw 32-bit floats for PractRand."""
#     with open(filename, "wb") as f:
#         for num in data:
#             f.write(struct.pack('f', num))
#     print(f"Saved binary data to {filename}")
