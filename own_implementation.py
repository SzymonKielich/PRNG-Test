import numpy as np
import math
import random
import scipy.stats as stats 

def LCG(a,c,m, seed, n):
    x_0 = seed
    random_numbers = []

    for _ in range(n):
        x_0 = (a*x_0 + c) % m
        random_numbers.append(x_0/m)

    return random_numbers

def chi_square_test(data, bins=10):
    observed_freq, _ = np.histogram(data, bins=bins)
    expected_freq = len(data) / bins

    chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)

    p_value = 1 - stats.chi2.cdf(chi2_stat, bins - 1)

    return chi2_stat, p_value

def ks_test(data):
    n = len(data)

    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, n+1) / n
    thereotical_cdf = sorted_data

    d_stat = np.max(np.abs(empirical_cdf - thereotical_cdf))

    p_value = np.exp(-2 * n * d_stat**2)

    return d_stat, p_value

def sum_test(data):
    sum_observed = np.sum(data)
    sum_expected = len(data) / 2

    dev = np.abs(sum_observed - sum_expected)

    var = len(data) / 12
    z_stat = dev / np.sqrt(var)

    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return z_stat, p_value

def d2_test(data):
    if len(data) % 2 != 0:
        data = data[:-1]
    
    x = np.array(data[::2])
    y = np.array(data[1::2])

    distances_sq = (x - 0.5)**2 + (y - 0.5)**2
    mean_d2 = np.mean(distances_sq)

    expected = 1/6  
    var = 1/(180 * len(distances_sq))  

    z_score = (mean_d2 - expected) / math.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return mean_d2, p_value


def collision_test(data, m=1000):
    n = len(data)
    hits = np.zeros(m, dtype=int)

    for x in data:
        idx = int(x * m)
        if idx == m:
            idx -= 1
        hits[idx] += 1

    collisions = np.sum(hits > 1)

    expected_collisions = m * (1 - ((1 - 1/m)**n + (n/m) * (1 - 1/m)**(n-1)))

    return collisions, expected_collisions

def runs_test(data):
    
    median = np.median(data)
    signs = [1 if x > median else 0 for x in data]

    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1

    n1 = np.sum(signs)
    n2 = len(signs) - n1

    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2)**2) * (n1 + n2 - 1))

    z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z, p_value