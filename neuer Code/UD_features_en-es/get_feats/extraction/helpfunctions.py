#! /usr/bin/python3
# coding: utf-8

"""
this sctipt contains only the lang-independent functions for translationese feature extraction with mega_collector.py
"""
import os
from collections import OrderedDict
from operator import itemgetter
import warnings

warnings.simplefilter("ignore")

lists_path = 'searchlists/'


def get_meta(input):
    # prepare for writing metadata
    lang_folder = len(os.path.abspath(input).split('/')) - 1
    status_folder = len(os.path.abspath(input).split('/')) - 2
    register_folder = len(os.path.abspath(input).split('/')) - 3
    
    status0 = os.path.abspath(input).split('/')[status_folder]
    register0 = os.path.abspath(input).split('/')[register_folder]
    
    lang0 = os.path.abspath(input).split('/')[lang_folder]
    
    return lang0, register0, status0


def get_trees(data, minlen=None):  # data is one object: a text or all of corpus as one file
    sentences0 = []
    badsents = 0
    only_punct = []
    current_sentence = []
    for line in data:
        if line.strip() == '':
            if current_sentence:
                sentences0.append(current_sentence)
            
            current_sentence = []
            only_punct = []
            continue

        if line.strip().startswith('#'):
            continue

        res = line.strip().split('\t')
        (identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
        if '.' in identifier or '-' in identifier:  # ignore empty nodes possible in the enhanced representations
            continue

        only_punct.append(res[3])
        var = list(set(only_punct))

        # throw away sentences consisting of punctuation marks only (ex. '.)')
        # and of numeral and a punctuation (ex. '3.', 'II.')
        if len(var) == 1 and var[0] == 'PUNCT':
            badsents += 1
            continue

        if len(var) == 2 and 'PUNCT' in var and 'NUM' in var:
            badsents += 1
        else:
            current_sentence.append((int(identifier), token, lemma, upos, xpos, feats, int(head), rel))
    
    if current_sentence:
        sentences0.append(current_sentence)
    
    sentences = [s for s in sentences0 if len(s) >= minlen]
    shorts = len(sentences0) - len(sentences)
    
    return sentences, badsents, shorts


# functions to traverse the trees
def get_headwd(node, sentence):  # when calling, test whether head exists --- if head:
    head_word = None
    head_id = node[6]
    
    for word in sentence:
        if head_id == word[0]:
            head_word = word
    return head_word


def get_kids(node, sentence):
    kids = []
    own_id = node[0]
    
    for word in sentence:
        if own_id == word[6]:
            kids.append(word)
    return kids  # requires iteration of children to get info on individual properties


def choose_kid_by_featrel(node, sentence, feat, rel):
    targetedkid_ind = None
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[7] == rel and feat in kid[5]:
            if sentence[0][0] == 2:
                '''
                this is a systemic error: in ru-data I have 7% of sentences that start with 2, instead of 1 which screws indexing
                ex. Нападения на иностранных студентов: tree[1] gets 'на' which has id=3, but in a zero-based Python tree-list is 1
                '''
                targetedkid_ind = kid[0] - 2
            else:
                targetedkid_ind = kid[0] - 1  # to be used as tree[targetedkid_ind]
    return targetedkid_ind


def choose_kid_by_posfeat(node, sentence, pos, feat):
    targetedkid_ind = None
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[3] == pos and feat in kid[5]:
            if sentence[0][0] == 2:
                '''
                this is a systemic error: in ru-data I have 7% of sentences that start with 2, instead of 1 which screws indexing
                ex. Нападения на иностранных студентов: tree[1] gets 'на' which has id=3, but in a zero-based Python tree-list is 1
                '''
                targetedkid_ind = kid[0] - 2
            else:
                targetedkid_ind = kid[0] - 1  # to be used as tree[targetedkid_ind]
    return targetedkid_ind


def choose_kid_by_posrel(node, sentence, pos, rel):
    targetedkid_ind = None
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[3] == pos and rel in kid[7]:
            # -1 is needed because tree[id] is refering to a 0-based list of words!
            # if no -1, I get: true_id=17, tree[true_id]==(18, ':', ':', 'PUNCT', ':', '_', 14, 'punct')
            if sentence[0][0] == 2:
                '''
                this is a systemic error: in ru-data I have 7% of sentences that start with 2, instead of 1 which screws indexing
                ex. Нападения на иностранных студентов: tree[1] gets 'на' which has id=3, but in a zero-based Python tree-list is 1
                '''
                targetedkid_ind = kid[0] - 2
            else:
                targetedkid_ind = kid[0] - 1  # to be used as tree[targetedkid_ind]
    return targetedkid_ind


def choose_kid_by_lempos(node, sentence, lemma, pos):
    targetedkid_ind = None
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[3] == pos and kid[2] == lemma:
            if sentence[0][0] == 2:
                '''
                this is a systemic error: in ru-data I have 7% of sentences that start with 2, instead of 1 which screws indexing
                ex. Нападения на иностранных студентов: tree[1] gets 'на' which has id=3, but in a zero-based Python tree-list is 1
                '''
                targetedkid_ind = kid[0] - 2
            else:
                targetedkid_ind = kid[0] - 1  # to be used as tree[targetedkid_ind]
    return targetedkid_ind


def has_auxkid_by_lem(node, sentence, lemma):
    res = False
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[3] == 'AUX' and kid[2] == lemma:
            res = True


def has_kid_by_lemlist(node, sentence, lemmas):
    res = False
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[2] in lemmas:
            res = True
    return res


def has_auxkid_by_tok(node, sentence, token):
    res = False
    kids = get_kids(node, sentence)
    for kid in kids:
        # specify kids features
        if kid[3] == 'AUX' and kid[1] == token:
            res = True
    
    return res  # test if True or False


# list of dependents pos
def get_kids_pos(node, sentence):  # there are no native tags (XPOS) in Russian corpora
    kids_pos = []
    own_id = node[0]
    for word in sentence:
        if own_id == word[6]:
            kids_pos.append(word[3])
    return kids_pos


# list of dependents pos
def get_kids_xpos(node, sentence):  # there are no native tags (XPOS) in Russian corpora
    kids_xpos = []
    own_id = node[0]
    for word in sentence:
        if own_id == word[6]:
            kids_xpos.append(word[4])
    return kids_xpos


# list of dependents dependency relations to the node
def get_kids_rel(node, sentence):
    kids_rel = []
    own_id = node[0]
    for word in sentence:
        if own_id == word[6]:
            kids_rel.append(word[7])
    return kids_rel


# flattened list of grammatical values; use with care in cases where there are many dependents
# -- gr feature can be on some other dependent
def get_kids_feats(node, sentence):
    deps_feats0 = []
    deps_feats1 = []
    deps_feats = []
    own_id = node[0]
    for word in sentence:
        if own_id == word[6]:
            deps_feats0.append(word[5])
        # split the string of several features and flatten the list
        for el in deps_feats0:
            try:
                el_lst = el.split('|')
                deps_feats1.append(el_lst)
            except:
                deps_feats1.append(el)
        deps_feats = [y for x in deps_feats1 for y in x]  # flatten the list
    return deps_feats


# list of dependents lemmas
def get_kids_lem(node, sentence):
    kids_lem = []
    own_id = node[0]
    for word in sentence:
        if own_id == word[6]:
            kids_lem.append(word[2])
    return kids_lem


def get_prev(node, sentence):
    prev = None
    for i, w in enumerate(sentence):
        if w == node:
            node_id = i
            prev = sentence[node_id - 1]
    
    return prev

###########################################################################
# file-level counts for normalisation


def wordcount(trees):
    words = 0
    for tree in trees:
        words += len(tree)
    
    return words


# this function adjusts counts for number of sentences, accounting for ca. 4% of errors in EN/GE
# where sentence ends with colon or semi-colon; in RU this error makes only 0.4%
def sents_num(trees, lang):
    sentnum = 0
    if lang == 'en':
        for tree in trees:
            lastwd = tree[-1]
            if not lastwd[2] in [':', ';', 'Mr.', 'Dr.']:
                sentnum += 1
    if lang == 'es':
        for tree in trees:
            lastwd = tree[-1]
            if not lastwd[2] in [':', ';']:
                sentnum += 1  # this is a fair num-of-sents count for a file
    
    return sentnum


def verbs_num(trees):
    verbs = 0
    for tree in trees:
        for w in tree:
            if w[3] == 'VERB':
                verbs += 1
    return verbs


def freqs_dic(trees, func, lang):
    dic = {}
    tot = 0
    for tree in trees:
        intree, lst = func(tree, lang)
        tot += intree
        for i in set(lst):
            freq = lst.count(i)
            if i in dic:
                dic[i] += freq
            else:
                dic[i] = freq
    
    dic_sort = OrderedDict(sorted(dic.items(), key=itemgetter(1), reverse=True))
    # print(list(dic_sort.items())[:100])
    tuples = list(dic_sort.items())[:100]
    for tu in tuples:
        print(':'.join(i for i in [tu[0], str(tu[1])]), end="; ")
    print("Dict size", len(list(dic_sort.items())))
    
    return tot


def support_all_lang(lang, lists_path=None):
    
    if lang == 'en':
        
        file0 = open(lists_path + "en_deverbals_stop.lst", 'r').readlines()
        pseudo_deverbs = []
        for wd in file0:
            wd = wd.strip()
            pseudo_deverbs.append(wd)
        
        file1 = open(lists_path + "en_adv_quantifiers.lst", 'r').readlines()
        quantifiers = []
        for q in file1:
            q = q.strip()
            quantifiers.append(q)
        
        file2 = open(lists_path + "en_modal-adj_predicates.lst", 'r').readlines()
        adj_pred = []
        for adj in file2:
            adj = adj.strip()
            adj_pred.append(adj)
        
        file3 = open(lists_path + "en_converts.lst", 'r').readlines()
        converts = []
        for conv in file3:
            conv = conv.strip()
            converts.append(conv)
    
    elif lang == 'de':
        file0 = lists_path + "de_deverbals_stop.lst"
        stoplist = open(file0, 'r').readlines()
        pseudo_deverbs = []
        for wd in stoplist:
            wd = wd.strip()
            pseudo_deverbs.append(wd)
        
        file1 = open(lists_path + "de_adv_quantifiers.lst", 'r').readlines()
        quantifiers = []
        for q in file1:
            q = q.strip()
            quantifiers.append(q)
        
        file3 = open(lists_path + "de_modal-adj_predicates.lst", 'r').readlines()
        adj_pred = []
        for adj in file3:
            adj = adj.strip()
            adj_pred.append(adj)
        
        file4 = open(lists_path + "de_converts.lst", 'r').readlines()
        converts = []
        for conv in file4:
            conv = conv.strip()
            converts.append(conv)
    
    elif lang == 'ru':
        
        file0 = lists_path + "ru_deverbals_stop.lst"
        stoplist = open(file0, 'r').readlines()
        pseudo_deverbs = []
        for wd in stoplist:
            wd = wd.strip()
            pseudo_deverbs.append(wd)
        
        file1 = open(lists_path + "ru_adv_quantifiers.lst", 'r').readlines()
        quantifiers = []
        for q in file1:
            q = q.strip()
            quantifiers.append(q)
        
        file2 = open(lists_path + "ru_modal-adj_predicates.lst", 'r').readlines()
        adj_pred = []
        for adj in file2:
            adj = adj.strip()
            adj_pred.append(adj)
        converts = []
        
    else:
        print(type(lang), lang)
        quantifiers = None
        adj_pred = None
        pseudo_deverbs = None
        converts = None
    
    return quantifiers, adj_pred, pseudo_deverbs, converts


def dms_support_all_langs(lang, lists_path=None):
    add = lists_path + lang + "_additive.lst"
    add_list = [i.strip() for i in open(add, 'r').readlines()]
    
    adv = lists_path + lang + "_adversative.lst"
    adv_list = [i.strip() for i in open(adv, 'r').readlines()]
    
    caus = lists_path + lang + "_causal.lst"
    caus_list = [i.strip() for i in open(caus, 'r').readlines()]
    
    temp_sequen = lists_path + lang + "_temp_sequen.lst"
    temp_sequen_list = [i.strip() for i in open(temp_sequen, 'r').readlines()]
    
    epistem = lists_path + lang + "_epistemic.lst"
    epist_list = [i.strip() for i in open(epistem, 'r').readlines()]
    
    return add_list, adv_list, caus_list, temp_sequen_list, epist_list