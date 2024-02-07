'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here
eps = 0.001

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    ## collect all words and tags in train set
    whole_dict = {}         # a dict {word : {tag : freq}}
    whole_tag_dict = {}     # a dict {tag : freq}, use to find tag for unseen dataset
    for sentence in train:
        for pair in sentence:
            if pair[1] in list(whole_tag_dict.keys()):
                whole_tag_dict[pair[1]] += 1
            else:
                whole_tag_dict[pair[1]] = 1
            try:
                tag_dict = whole_dict[pair[0]]
                if pair[1] in list(tag_dict.keys()):
                    tag_dict[pair[1]] += 1
                else:
                    tag_dict[pair[1]] = 1
            except:
                tag_dict = {}
                tag_dict[pair[1]] = 1
                whole_dict[pair[0]] = tag_dict

    ## set tag for unseen dataset
    whole_tag_list = sorted(whole_tag_dict.items(), key=lambda x: x[1], reverse=True)
    tag_unseen = whole_tag_list[0][0]

    ## keep only words and tag
    check_dict = {}
    for word in list(whole_dict.keys()):
        tag_dict = whole_dict[word]
        tag_list = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)
        check_dict[word] = tag_list[0][0]

    ## match on test set
    result = []
    for sentence in test:
        sentence_pred = []
        for test_word in sentence:
            try:
                tag_pred = check_dict[test_word]
            except:
                tag_pred = tag_unseen
            sentence_pred.append((test_word, tag_pred))
        result.append(sentence_pred)

    return result

def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    import numpy as np

    ## Step 1: Count occurrences of tags, tag pairs, tag/word pairs
    # collect all words and tags in train set
    t_dict = {}         # tag freq dict {tag : freq}
    tw_dict = {}        # word given tag prob dict {tag : {word : freq}}
    tt_dict = {}        # tag given previous tag prob dict {tag(prev) : {tag(cur) : freq}}
    for sentence in train:
        prev_tag = 0
        for pair in sentence:
            # update t_dict
            try:
                t_dict[pair[1]] += 1
            except:
                t_dict[pair[1]] = 1
            # upate tw_dict
            if pair[1] in list(tw_dict.keys()):
                word_dict = tw_dict[pair[1]]
                try:
                    word_dict[pair[0]] += 1
                except:
                    word_dict[pair[0]] = 1
            else:
                word_dict = {}
                word_dict[pair[0]] = 1
                tw_dict[pair[1]] = word_dict
            # update tt_dict
            if prev_tag != 0:          # has previous tag
                if prev_tag in list(tt_dict.keys()):
                    tag_dict = tt_dict[prev_tag]
                    try:
                        tag_dict[pair[1]] += 1
                    except:
                        tag_dict[pair[1]] = 1
                else:
                    tag_dict = {}
                    tag_dict[pair[1]] = 1
                    tt_dict[prev_tag] = tag_dict
            prev_tag = pair[1]

    ## Step 2: Compute smoothed probabilities
    t_prob_dict = {}                # {tag : prob}
    tw_prob_dict = {}               # {tag : {word : prob}}
    tt_prob_dict = {}               # {tag(prev) : {tag(cur) : prob}}
    # build t_prob_dict
    denom_t = sum(list(t_dict.values())) + eps * (len(list(t_dict.keys())) + 1)
    for tag in list(t_dict.keys()):
        t_prob_dict[tag] = (t_dict[tag] + eps) / denom_t
    t_prob_dict["UNKNOWN"] = eps / denom_t
    # build tw_prob_dict
    for tag in list(tw_dict.keys()):
        word_dict = {}
        denom_tw = sum(list(tw_dict[tag].values())) + eps * (len(list(tw_dict[tag].keys())) + 1)
        for word in list(tw_dict[tag].keys()):
            word_dict[word] = (tw_dict[tag][word] + eps) / denom_tw
        word_dict["UNKNOWN"] = eps / denom_tw
        tw_prob_dict[tag] = word_dict
    # build tt_prob_dict
    for tag_prev in list(tt_dict.keys()):
        tag_dict = {}
        denom_tt = sum(list(tt_dict[tag_prev].values())) + eps * (len(list(tt_dict[tag_prev].keys())) + 1)
        for tag in list(tt_dict[tag_prev].keys()):
            tag_dict[tag] = (tt_dict[tag_prev][tag] + eps) / denom_tt
        tag_dict["UNKNOWN"] = eps / denom_tt
        tt_prob_dict[tag_prev] = tag_dict

    ## Step 3: Take the log of each probability
    for tag in list(t_prob_dict.keys()):
        t_prob_dict[tag] = float(np.log(t_prob_dict[tag]))
    for tag in list(tw_prob_dict.keys()):
        for word in tw_prob_dict[tag]:
            tw_prob_dict[tag][word] = float(np.log(tw_prob_dict[tag][word]))
    for tag_prev in list(tt_prob_dict.keys()):
        for tag in tt_prob_dict[tag_prev]:
            tt_prob_dict[tag_prev][tag] = float(np.log(tt_prob_dict[tag_prev][tag]))

    ## Step 4: Construct the trellis
    result = []
    for sentence in test:
        time = 0
        bptr_dict = {}              # back trace pointer dict {time : {tag(cur) : tag(prev)}}
        value_dict = {}             # state value dict {time: {tag(cur) : value}}
        sentence_pred = []
        for word in sentence:
            tag_v_dict = {}
            tag_p_dict = {}
            if word == "START":          # start point
                sentence_pred.append((word, "START"))
                for tag in list(t_prob_dict.keys()):
                    if tag == "START":
                        try:
                            value_dict[time][tag] = 10000
                        except:
                            value_dict[time] = {tag : 10000}
                    else:
                        try:
                            value_dict[time][tag] = 0
                        except:
                            value_dict[time] = {tag : 0}
            elif word != "END":         # not stop point
                for tag in list(tw_prob_dict.keys()):
                    value_opt = "nothing"                               # optimal value to be stored
                    prev_opt = 0                                        # optimal previous tag to be stored
                    for prev_tag in list(tt_prob_dict.keys()):
                        value = value_dict[time-1][prev_tag]            # previous state value
                        try:
                            value += tt_prob_dict[prev_tag][tag]            # Transition
                        except:
                            value += tt_prob_dict[prev_tag]["UNKNOWN"]      # Transition
                        try:
                            value += tw_prob_dict[tag][word]                # Emission
                        except:
                            value += tw_prob_dict[tag]["UNKNOWN"]           # Emission

                        if value_opt == "nothing":                      # not initialize yet
                            value_opt = value
                            prev_opt = prev_tag
                        elif value > value_opt:
                            value_opt = value
                            prev_opt = prev_tag
                    tag_v_dict[tag] = value_opt                   # store value of current state
                    tag_p_dict[tag] = prev_opt
                value_dict[time] = tag_v_dict
                bptr_dict[time] = tag_p_dict
            time += 1

        ## Step 5: Back trace
        # process end point
        time -= 2                               # point to time before end
        value_opt = "nothing"                   # optimal value to be stored
        prev_opt = 0                            # optimal previous tag to be stored
        for tag_prev in list(tt_prob_dict.keys()):
            value = value_dict[time][tag_prev]              # previous state value
            try:
                value += tt_prob_dict[tag_prev]["END"]          # Transition
            except:
                value += tt_prob_dict[tag_prev]["UNKNOWN"]          # Transition
            if value_opt == "nothing":          # not initialize yet
                value_opt = value
                prev_opt = tag_prev
            elif value > value_opt:
                value_opt = value
                prev_opt = tag_prev
        # choose along path
        back_path = []
        while time > 0:
            back_path.append((sentence[time], prev_opt))
            prev_opt = bptr_dict[time][prev_opt]
            time -= 1

        ## Step 6: generate result
        back_path.reverse()
        sentence_pred += back_path
        sentence_pred.append((sentence[-1], "END"))

        result.append(sentence_pred)

    return result


def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    smoothness = 0.00001

    import numpy as np

    ## Step 1: Count occurrences of tags, tag pairs, tag/word pairs
    # collect all words and tags in train set
    t_dict = {}  # tag freq dict {tag : freq}
    tw_dict = {}  # word given tag prob dict {tag : {word : freq}}
    tt_dict = {}  # tag given previous tag prob dict {tag(prev) : {tag(cur) : freq}}
    word_freq_dict = {}         # word and frequency {word : [tag, freq]}
    for sentence in train:
        prev_tag = 0
        for pair in sentence:
            # update word_freq_dict
            try:
                word_freq_dict[pair[0]][1] += 1
            except:
                word_freq_dict[pair[0]] = [pair[1], 1]
            # update t_dict
            try:
                t_dict[pair[1]] += 1
            except:
                t_dict[pair[1]] = 1
            # upate tw_dict
            if pair[1] in list(tw_dict.keys()):
                word_dict = tw_dict[pair[1]]
                try:
                    word_dict[pair[0]] += 1
                except:
                    word_dict[pair[0]] = 1
            else:
                word_dict = {}
                word_dict[pair[0]] = 1
                tw_dict[pair[1]] = word_dict
            # update tt_dict
            if prev_tag != 0:  # has previous tag
                if prev_tag in list(tt_dict.keys()):
                    tag_dict = tt_dict[prev_tag]
                    try:
                        tag_dict[pair[1]] += 1
                    except:
                        tag_dict[pair[1]] = 1
                else:
                    tag_dict = {}
                    tag_dict[pair[1]] = 1
                    tt_dict[prev_tag] = tag_dict
            prev_tag = pair[1]

    ## Step 2: Compute P(T|hapax)
    hapax_tag_dict = {}             # {tag : freq}
    for word in list(word_freq_dict.keys()):
        if word_freq_dict[word][1] == 1:               # hapax
            try:
                hapax_tag_dict[word_freq_dict[word][0]] += 1
            except:
                hapax_tag_dict[word_freq_dict[word][0]] = 1
    hapax_tag_prob_dict = {}
    hapax_num = sum(list(hapax_tag_dict.values()))
    for tag in list(hapax_tag_dict.keys()):
        hapax_tag_prob_dict[tag] = hapax_tag_dict[tag] / hapax_num
    print(hapax_tag_prob_dict)

    ## Step 3: Compute smoothed probabilities
    t_prob_dict = {}  # {tag : prob}
    tw_prob_dict = {}  # {tag : {word : prob}}
    tt_prob_dict = {}  # {tag(prev) : {tag(cur) : prob}}
    # build t_prob_dict
    denom_t = sum(list(t_dict.values())) + eps * (len(list(t_dict.keys())) + 1)
    for tag in list(t_dict.keys()):
        t_prob_dict[tag] = (t_dict[tag] + eps) / denom_t
    t_prob_dict["UNKNOWN"] = eps / denom_t
    # build tw_prob_dict
    for tag in list(tw_dict.keys()):
        word_dict = {}
        try:
            denom_tw = sum(list(tw_dict[tag].values())) + smoothness * hapax_tag_prob_dict[tag] * (len(list(tw_dict[tag].keys())) + 1)
            for word in list(tw_dict[tag].keys()):
                word_dict[word] = (tw_dict[tag][word] + smoothness * hapax_tag_prob_dict[tag]) / denom_tw
            word_dict["UNKNOWN"] = smoothness * hapax_tag_prob_dict[tag] / denom_tw
        except:
            denom_tw = sum(list(tw_dict[tag].values())) + smoothness * smoothness * (
                        len(list(tw_dict[tag].keys())) + 1)
            for word in list(tw_dict[tag].keys()):
                word_dict[word] = (tw_dict[tag][word] + smoothness * smoothness) / denom_tw
            word_dict["UNKNOWN"] = smoothness * smoothness / denom_tw
        tw_prob_dict[tag] = word_dict
    # build tt_prob_dict
    for tag_prev in list(tt_dict.keys()):
        tag_dict = {}
        denom_tt = sum(list(tt_dict[tag_prev].values())) + eps * (len(list(tt_dict[tag_prev].keys())) + 1)
        for tag in list(tt_dict[tag_prev].keys()):
            tag_dict[tag] = (tt_dict[tag_prev][tag] + eps) / denom_tt
        tag_dict["UNKNOWN"] = eps / denom_tt
        tt_prob_dict[tag_prev] = tag_dict

    ## Step 3: Take the log of each probability
    for tag in list(t_prob_dict.keys()):
        t_prob_dict[tag] = float(np.log(t_prob_dict[tag]))
    for tag in list(tw_prob_dict.keys()):
        for word in tw_prob_dict[tag]:
            tw_prob_dict[tag][word] = float(np.log(tw_prob_dict[tag][word]))
    for tag_prev in list(tt_prob_dict.keys()):
        for tag in tt_prob_dict[tag_prev]:
            tt_prob_dict[tag_prev][tag] = float(np.log(tt_prob_dict[tag_prev][tag]))

    ## Step 4: Construct the trellis
    result = []
    for sentence in test:
        time = 0
        bptr_dict = {}  # back trace pointer dict {time : {tag(cur) : tag(prev)}}
        value_dict = {}  # state value dict {time: {tag(cur) : value}}
        sentence_pred = []
        for word in sentence:
            tag_v_dict = {}
            tag_p_dict = {}
            if word == "START":  # start point
                sentence_pred.append((word, "START"))
                for tag in list(t_prob_dict.keys()):
                    if tag == "START":
                        try:
                            value_dict[time][tag] = 10000
                        except:
                            value_dict[time] = {tag: 10000}
                    else:
                        try:
                            value_dict[time][tag] = 0
                        except:
                            value_dict[time] = {tag: 0}
            elif word != "END":  # not stop point
                for tag in list(tw_prob_dict.keys()):
                    value_opt = "nothing"  # optimal value to be stored
                    prev_opt = 0  # optimal previous tag to be stored
                    for prev_tag in list(tt_prob_dict.keys()):
                        value = value_dict[time - 1][prev_tag]  # previous state value
                        try:
                            value += tt_prob_dict[prev_tag][tag]  # Transition
                        except:
                            value += tt_prob_dict[prev_tag]["UNKNOWN"]  # Transition
                        try:
                            value += tw_prob_dict[tag][word]  # Emission
                        except:
                            value += tw_prob_dict[tag]["UNKNOWN"]  # Emission

                        if value_opt == "nothing":  # not initialize yet
                            value_opt = value
                            prev_opt = prev_tag
                        elif value > value_opt:
                            value_opt = value
                            prev_opt = prev_tag
                    tag_v_dict[tag] = value_opt  # store value of current state
                    tag_p_dict[tag] = prev_opt
                value_dict[time] = tag_v_dict
                bptr_dict[time] = tag_p_dict
            time += 1

        ## Step 5: Back trace
        # process end point
        time -= 2  # point to time before end
        value_opt = "nothing"  # optimal value to be stored
        prev_opt = 0  # optimal previous tag to be stored
        for tag_prev in list(tt_prob_dict.keys()):
            value = value_dict[time][tag_prev]  # previous state value
            try:
                value += tt_prob_dict[tag_prev]["END"]  # Transition
            except:
                value += tt_prob_dict[tag_prev]["UNKNOWN"]  # Transition
            if value_opt == "nothing":  # not initialize yet
                value_opt = value
                prev_opt = tag_prev
            elif value > value_opt:
                value_opt = value
                prev_opt = tag_prev
        # choose along path
        back_path = []
        while time > 0:
            back_path.append((sentence[time], prev_opt))
            prev_opt = bptr_dict[time][prev_opt]
            time -= 1

        ## Step 6: generate result
        back_path.reverse()
        sentence_pred += back_path
        sentence_pred.append((sentence[-1], "END"))

        result.append(sentence_pred)

    return result



