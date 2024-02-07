'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    counts = np.zeros(len(texts))           # a list of integers indicating number of word0 in each text
    for textIdx in range(len(texts)):
        counts[textIdx] = texts[textIdx].count(word0)
    cX0 = int(np.sort(counts)[-1] + 1)           # largest possible times of word0 show up in a single text + 1
    Pmarginal = np.zeros(cX0)
    for x0 in range(cX0):
        Pmarginal[x0] = np.sum(counts == x0) / len(texts)
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    counts0 = np.zeros(len(texts))          # a list of integers indicating number of word0 in each text
    counts1 = np.zeros(len(texts))          # a list of integers indicating number of word1 in each text
    for textIdx in range(len(texts)):
        counts0[textIdx] = texts[textIdx].count(word0)
        counts1[textIdx] = texts[textIdx].count(word1)
    cX0 = int(np.sort(counts0)[-1] + 1)             # largest possible times of word0 show up in a single text + 1
    cX1 = int(np.sort(counts1)[-1] + 1)             # largest possible times of word1 show up in a single text + 1
    Pcond = np.zeros((cX0, cX1))
    for x0 in range(cX0):
        for x1 in range(cX1):
            count_both = 0                          # count how many texts both meet x0, x1
            for textIdx in range(len(texts)):
                if counts0[textIdx] == x0 and counts1[textIdx] == x1:
                    count_both += 1
            Pcond[x0][x1] = count_both / np.sum(counts0 == x0)
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    cX0 = np.shape(Pcond)[0]
    cX1 = np.shape(Pcond)[1]
    Pjoint = Pcond
    for x0 in range(cX0):
        for x1 in range(cX1):
            if Pcond[x0][x1] <= 1:              # conditional prob is not np.nan
                Pjoint[x0][x1] = Pcond[x0][x1] * Pmarginal[x0]
            else:                               # conditional prob is np.nan
                Pjoint[x0][x1] = 0
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    X0_dis = np.sum(Pjoint, axis=1)             # X0 distribution
    X1_dis = np.sum(Pjoint, axis=0)             # X1 distribution
    X0_mean = 0
    X1_mean = 0
    for value0 in range(int(np.shape(X0_dis)[0])):
        X0_mean += value0 * X0_dis[value0]
    for value1 in range(int(np.shape(X1_dis)[0])):
        X1_mean += value1 * X1_dis[value1]
    mu = np.zeros(2)
    mu[0] = X0_mean
    mu[1] = X1_mean
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    X0_mean = mu[0]             # E[X]
    X1_mean = mu[1]             # E[Y]
    X0_dis = np.sum(Pjoint, axis=1)         # X0 distribution
    X1_dis = np.sum(Pjoint, axis=0)         # X1 distribution
    X01_mean = 0                # E[XY]
    for x0 in range(int(np.shape(Pjoint)[0])):
        for x1 in range(int(np.shape(Pjoint)[1])):
            X01_mean += (x0 * x1) * Pjoint[x0][x1]
    X0_var = 0                  # Var(X)
    X1_var = 0                  # Var(Y)
    for x0 in range(int(np.shape(X0_dis)[0])):
        X0_var += (x0 - X0_mean) ** 2 * X0_dis[x0]
    for x1 in range(int(np.shape(X1_dis)[0])):
        X1_var += (x1 - X1_mean) ** 2 * X1_dis[x1]
    Sigma = np.zeros((2, 2))
    Sigma[0][0] = X0_var
    Sigma[1][1] = X1_var
    Sigma[0][1] = X01_mean - X0_mean * X1_mean          # Cov(X,Y) = E[XY] - E[X]E[Y]
    Sigma[1][0] = X01_mean - X0_mean * X1_mean
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    import collections
    Pfunc = collections.Counter()
    for x0 in range(int(np.shape(Pjoint)[0])):
        for x1 in range(int(np.shape(Pjoint)[1])):
            z = f(x0, x1)
            Pfunc[z] += Pjoint[x0][x1]
    return Pfunc
    
