# Treatment T, Sex S, Cancer C, Death D. All binary, all Bernoulli

## WE SET
# P(S=1) = 0.5
# P(C=1) = 0.3


import numpy as np
import pandas as pd
import scipy.stats as stats


def generate_natural_exp_data(threshold=55, sd=0.1):

    N = 20000
    p_sex, mu_age = 0.5, 50

    S = np.random.binomial(1, p_sex, N)
    A = np.random.normal(loc=mu_age, scale=20, size=N)
    T, D = [], []
    y1, y0 = [], []

    # conditional treatment probabilities for sex/cancer values
    p11, p10 = 0.5, 0.3
    p01, p00 = 0.1, 0.4


    # conditional death probabilities for the treated
    p11_t, p10_t, p01_t, p00_t = 0.1, 0.2, 0.4, 0.15
    # for the untreated
    p11_u, p10_u, p01_u, p00_u = 0.2, 0.4, 0.8, 0.3

    for i in range(N):
        if S[i]==1 and A[i]>= threshold:
            propensity = stats.truncnorm.rvs((0 - p11) / sd, (1 - p11) / sd, p11, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 1 and A[i] < threshold:
            propensity = stats.truncnorm.rvs((0 - p10) / sd, (1 - p10) / sd, p10, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 0 and A[i] >= threshold:
            propensity = stats.truncnorm.rvs((0 - p01) / sd, (1 - p01) / sd, p01, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        else:
            propensity = stats.truncnorm.rvs((0 - p00) / sd, (1 - p00) / sd, p00, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)

    for i in range(N):
        if T[i]==1:
            if S[i]==1 and A[i] >= threshold:
                D.append(np.random.binomial(1, p11_t))
                y0.append(np.random.binomial(1, p11_u)) #potential outcomes D(T=0)
            elif S[i] == 1 and A[i] < threshold:
                D.append(np.random.binomial(1, p10_t))
                y0.append(np.random.binomial(1, p10_u))
            elif S[i] == 0 and A[i] >= threshold:
                D.append(np.random.binomial(1, p01_t))
                y0.append(np.random.binomial(1, p01_u))
            else:
                D.append(np.random.binomial(1, p00_t))
                y0.append(np.random.binomial(1, p00_u))
            y1.append(D[i]) # potential outcomes D(T=1) = observed D
        else:
            if S[i] == 1 and A[i] >= threshold:
                D.append(np.random.binomial(1, p11_u))
                y1.append(np.random.binomial(1, p11_t)) #potential outcomes D(T=1)
            elif S[i] == 1 and A[i] < threshold:
                D.append(np.random.binomial(1, p10_u))
                y1.append(np.random.binomial(1, p10_t))
            elif S[i] == 0 and A[i] >= threshold:
                D.append(np.random.binomial(1, p01_u))
                y1.append(np.random.binomial(1, p01_t))
            else:
                D.append(np.random.binomial(1, p00_u))
                y1.append(np.random.binomial(1, p11_t))
            y0.append(D[i]) # potential outcomes D(T=0) = observed D


    values = np.column_stack((S, A, T, D, y1, y0))
    data = pd.DataFrame(values, columns=['S','C','T','D', 'y1', 'y0'])
    return data
