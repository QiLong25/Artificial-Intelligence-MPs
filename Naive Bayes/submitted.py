'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''

    from collections import Counter

    pairs_pos = []
    pairs_neg = []
    ## create a list of pairs in pos dataset
    for i in range(len(train['pos'])):
        for k in range(len(train['pos'][i]) - 1):
            pair = train['pos'][i][k] + "*-*-*-*" + train['pos'][i][k+1]
            pairs_pos.append(pair)
    ## create a list of pairs in neg dataset
    for i in range(len(train['neg'])):
        for k in range(len(train['neg'][i]) - 1):
            pair = train['neg'][i][k] + "*-*-*-*" + train['neg'][i][k + 1]
            pairs_neg.append(pair)
    ## count frequency
    frequency = {}
    frequency['pos'] = Counter(pairs_pos)
    frequency['neg'] = Counter(pairs_neg)

    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''

    import copy

    nonstop = {}
    nonstop['pos'] = copy.deepcopy(frequency['pos'])
    nonstop['neg'] = copy.deepcopy(frequency['neg'])
    pairs_pos = list(nonstop['pos'].keys())
    pairs_neg = list(nonstop['neg'].keys())
    ## delete stopword pairs from pos class
    for text in pairs_pos:
        words = text.split("*-*-*-*")
        if words[0] in stopwords and words[1] in stopwords:
            del nonstop['pos'][text]
    ## delete stopword pairs from neg class
    for text in pairs_neg:
        words = text.split("*-*-*-*")
        if words[0] in stopwords and words[1] in stopwords:
            del nonstop['neg'][text]

    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''

    import numpy as np

    likelihood = {}
    likelihood['pos'] = {}
    likelihood['neg'] = {}
    total_bigram_pos = sum(nonstop['pos'].values())
    total_bitype_pos = len(nonstop['pos'].keys())
    total_bigram_neg = sum(nonstop['neg'].values())
    total_bitype_neg = len(nonstop['neg'].keys())

    ## smooth pos
    denom = total_bigram_pos + smoothness * (total_bitype_pos + 1)
    texts = list(nonstop['pos'].keys())
    likelihood_matrix = np.zeros(len(texts) + 1)
    for idx in range(len(texts)):
        text = texts[idx]
        likelihood_matrix[idx] = nonstop['pos'][text]
    # calculate smoothed values
    likelihood_matrix = (likelihood_matrix + smoothness) / denom
    # assign probability back to dict
    for idx in range(len(texts)):
        likelihood['pos'][texts[idx]] = float(likelihood_matrix[idx])
    likelihood['pos']['OOV'] = float(likelihood_matrix[len(texts)])
    ## smooth neg
    denom = total_bigram_neg + smoothness * (total_bitype_neg + 1)
    texts = list(nonstop['neg'].keys())
    likelihood_matrix = np.zeros(len(texts) + 1)
    for idx in range(len(texts)):
        text = texts[idx]
        likelihood_matrix[idx] = nonstop['neg'][text]
    likelihood_matrix = (likelihood_matrix + smoothness) / denom
    # assign probability back to dict
    for idx in range(len(texts)):
        likelihood['neg'][texts[idx]] = float(likelihood_matrix[idx])
    likelihood['neg']['OOV'] = float(likelihood_matrix[len(texts)])

    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''

    import numpy as np

    hypotheses = []
    pos_keys = list(likelihood['pos'].keys())
    pos_dict = likelihood['pos']
    neg_keys = list(likelihood['neg'].keys())
    neg_dict = likelihood['neg']
    pos_OOV = float(np.log(pos_dict['OOV']))
    neg_OOV = float(np.log(neg_dict['OOV']))
    pos_log_dict = {}
    neg_log_dict = {}
    # pos_log = np.ones(len(pos_keys))
    # neg_log = np.ones(len(neg_keys))
    for idx in range(len(pos_keys)):
        pos_key = pos_keys[idx]
        pos_log_dict[pos_key] = float(np.log(pos_dict[pos_key]))
    for idx in range(len(neg_keys)):
        neg_key = neg_keys[idx]
        neg_log_dict[neg_key] = float(np.log(neg_dict[neg_key]))

    for i in range(len(texts)):
        unit_text = texts[i]
        Ppos = float(np.log(prior))
        Pneg = float(np.log(1 - prior))
        for k in range(len(unit_text) - 1):
            if unit_text[k] in stopwords and unit_text[k+1] in stopwords:
                continue
            text = unit_text[k] + "*-*-*-*" + unit_text[k+1]
            try:
                Ppos += pos_log_dict[text]
            except:
                Ppos += pos_OOV
            try:
                Pneg += neg_log_dict[text]
            except:
                Pneg += neg_OOV
        if Ppos > Pneg:
            hypotheses.append("pos")
        elif Ppos < Pneg:
            hypotheses.append("neg")
        else:
            hypotheses.append("undecided")

    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''

    import numpy as np

    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for m in range(len(priors)):
        for n in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[n])
            hypotheses = naive_bayes(texts, likelihood, priors[m])
            correct = 0
            for i in range(len(hypotheses)):
                if hypotheses[i] == labels[i]:
                    correct += 1
            accuracies[m][n] = correct / len(texts)

    return accuracies
                          