# Treatment T, Sex S, Cancer C, A Arrythmia, Death D. All binary, all Bernoulli

## WE SET
# P(S=1) = 0.5
# P(C=1) = 0.3
# P(A=1) = 0.1

# Setting the positivity violations in two children nodes
# Node { S=0, C=0, A=0 } has a noisy probability 0 of treatment
# Node { S=1, C=1, A=1 } has a noisy probability 1 of treatment


import numpy as np
import pandas as pd
import scipy.stats as stats


def generate_pos_violations_data(sd=0.1):

    N = 20000
    p_sex, p_cancer, p_arythmic = 0.5, 0.3, 0.1

    S = np.random.binomial(1, p_sex, N)
    C = np.random.binomial(1, p_cancer, N)
    A = np.random.binomial(1, p_arythmic, N)
    T, D = [], []
    y1, y0 = [], []

    # conditional treatment probabilities for sex/cancer/arythmic values
    p101, p110, p100 = 0.32, 0.12, 0.42
    p010, p011 = 0.17, 0.3
    p001 = 0.24

    # conditional death probabilities for the treated
    p111_t, p110_t = 0.13, 0.08
    p101_t, p100_t = 0.21, 0.1
    p011_t, p010_t = 0.36, 0.29
    p001_t, p000_t = 0.24, 0.09

    # for the untreated
    p111_u, p110_u = 0.31, 0.40
    p101_u, p100_u = 0.29, 0.45
    p011_u, p010_u = 0.4, 0.51
    p001_u, p000_u = 0.43, 0.73

    for i in range(N):
        # positivity violation: all treated
        if S[i] == 1 and C[i] == 1 and A[i] == 1:
            propensity = stats.truncnorm.rvs((0 - 1) / (sd/5), (1 - 1) / (sd/5), 1, (sd/5))
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        # positivity violation: all untreated
        elif S[i] == 0 and C[i] == 0 and A[i] == 0:
            propensity = stats.truncnorm.rvs((0 - 0) / (sd/5), (1 - 0) / (sd/5), 0, (sd/5))
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 1 and C[i] == 0 and A[i] == 1:
            propensity = stats.truncnorm.rvs((0 - p101) / sd, (1 - p101) / sd, p101, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 1 and C[i] == 1 and A[i] == 0:
            propensity = stats.truncnorm.rvs((0 - p110) / sd, (1 - p110) / sd, p110, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 1 and C[i] == 0 and A[i] == 0:
            propensity = stats.truncnorm.rvs((0 - p100) / sd, (1 - p100) / sd, p100, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 0 and C[i] == 1 and A[i] == 0:
            propensity = stats.truncnorm.rvs((0 - p010) / sd, (1 - p010) / sd, p010, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        elif S[i] == 0 and C[i] == 1 and A[i] == 1:
            propensity = stats.truncnorm.rvs((0 - p011) / sd, (1 - p011) / sd, p011, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)
        else:
            propensity = stats.truncnorm.rvs((0 - p001) / sd, (1 - p001) / sd, p001, sd)
            value = np.random.binomial(1, propensity, 1)
            T.append(value)

    for i in range(N):
        if T[i] == 1:
            if S[i] == 1 and C[i] == 1 and A[i] == 1:
                D.append(np.random.binomial(1, p111_t))
                y0.append(np.random.binomial(1, p111_u))
            elif S[i] == 0 and C[i] == 0 and A[i] == 0:
                D.append(np.random.binomial(1, p000_t))
                y0.append(np.random.binomial(1, p000_u))
            elif S[i] == 1 and C[i] == 0 and A[i] == 1:
                D.append(np.random.binomial(1, p101_t))
                y0.append(np.random.binomial(1, p101_u))
            elif S[i] == 1 and C[i] == 1 and A[i] == 0:
                D.append(np.random.binomial(1, p110_t))
                y0.append(np.random.binomial(1, p110_u))
            elif S[i] == 1 and C[i] == 0 and A[i] == 0:
                D.append(np.random.binomial(1, p100_t))
                y0.append(np.random.binomial(1, p100_u))
            elif S[i] == 0 and C[i] == 1 and A[i] == 0:
                D.append(np.random.binomial(1, p010_t))
                y0.append(np.random.binomial(1, p010_u))
            elif S[i] == 0 and C[i] == 1 and A[i] == 1:
                D.append(np.random.binomial(1, p011_t))
                y0.append(np.random.binomial(1, p011_u))
            else:
                D.append(np.random.binomial(1, p001_t))
                y0.append(np.random.binomial(1, p001_u))
            y1.append(D[i])

        else:
            if S[i] == 1 and C[i] == 1 and A[i] == 1:
                D.append(np.random.binomial(1, p111_u))
                y1.append(np.random.binomial(1, p111_t))
            elif S[i] == 0 and C[i] == 0 and A[i] == 0:
                D.append(np.random.binomial(1, p000_u))
                y1.append(np.random.binomial(1, p000_t))
            elif S[i] == 1 and C[i] == 0 and A[i] == 1:
                D.append(np.random.binomial(1, p101_u))
                y1.append(np.random.binomial(1, p101_t))
            elif S[i] == 1 and C[i] == 1 and A[i] == 0:
                D.append(np.random.binomial(1, p110_u))
                y1.append(np.random.binomial(1, p110_t))
            elif S[i] == 1 and C[i] == 0 and A[i] == 0:
                D.append(np.random.binomial(1, p100_u))
                y1.append(np.random.binomial(1, p100_t))
            elif S[i] == 0 and C[i] == 1 and A[i] == 0:
                D.append(np.random.binomial(1, p010_u))
                y1.append(np.random.binomial(1, p010_t))
            elif S[i] == 0 and C[i] == 1 and A[i] == 1:
                D.append(np.random.binomial(1, p011_u))
                y1.append(np.random.binomial(1, p011_t))
            else:
                D.append(np.random.binomial(1, p001_u))
                y1.append(np.random.binomial(1, p001_t))
            y0.append(D[i])

    values = np.column_stack((S, C, A, T, D, y1, y0))
    data = pd.DataFrame(values, columns=['S','C', 'A', 'T','D', 'y1', 'y0'])
    return data
