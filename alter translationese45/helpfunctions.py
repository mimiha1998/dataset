#! /usr/bin/python3
# coding: utf-8

'''
this sctipt contains only the lang-independent functions
'''
import os
from igraph import Graph, ARPACKOptions
from collections import OrderedDict
from operator import itemgetter
import warnings
warnings.simplefilter("ignore")

base_path = 'C:/Users/Fox0197/Desktop/projekt_final//'
lists_path = 'c:/Users/Fox0197/Desktop/projekt_final/searchlists//'

def get_meta(input):
	# prepare for writing metadata
	lang_folder = len(os.path.abspath(input).split('/')) - 1
	status_folder = len(os.path.abspath(input).split('/')) - 2
	korp_folder = len(os.path.abspath(input).split('/')) - 3
	
	status0 = os.path.abspath(input).split('/')[status_folder]
	korp0 = os.path.abspath(input).split('/')[korp_folder]

	lang0 = os.path.abspath(input).split('/')[lang_folder]

	return lang0,korp0,status0

def get_trees(data): # data is one object: a text or all of corpus as one file
	sentences = []
	only_punct = []
	current_sentence = []
	for line in data:
		if line.strip() == '':
			if current_sentence:
				sentences.append(current_sentence)

			current_sentence = []
			only_punct = []

			continue
		if line.strip().startswith('#'):
			continue
		res = line.strip().split('\t')
		(identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
		if '.' in identifier or '-' in identifier:  # ignore empty nodes possible in the enhanced representations
			continue

		for i in res:
			only_punct.append(res[3])
		var = list(set(only_punct))
		# throw away sentences that consist of just PUNCT, particularly rare 4+ PUNCT
		if len(var) == 1 and var[0] == 'PUNCT':
			continue
		else:
			current_sentence.append((int(identifier), token, lemma, upos, xpos, feats, int(head), rel))

	if current_sentence:
		sentences.append(current_sentence)
		
	sentences = [s for s in sentences if len(s) >= 4]

	return sentences

## functions to traverse the trees
def get_headwd(node, sentence): # when calling, test whether head exists --- if head:
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
	return kids # requires iteration of children to get info on individual properties

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
				targetedkid_ind = kid[0]-1  # to be used as tree[targetedkid_ind]
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
				targetedkid_ind = kid[0]-1  # to be used as tree[targetedkid_ind]
	return targetedkid_ind

def choose_kid_by_posrel(node, sentence, pos, rel):
	targetedkid_ind = None
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[3] == pos and rel in kid[7]:
			## -1 is needed because tree[id] is refering to a 0-based list of words!
			## if no -1, I get: true_id=17, tree[true_id]==(18, ':', ':', 'PUNCT', ':', '_', 14, 'punct')
			if sentence[0][0] == 2:
				'''
				this is a systemic error: in ru-data I have 7% of sentences that start with 2, instead of 1 which screws indexing
				ex. Нападения на иностранных студентов: tree[1] gets 'на' which has id=3, but in a zero-based Python tree-list is 1
				'''
				targetedkid_ind = kid[0] - 2
			else:
				targetedkid_ind = kid[0]-1  # to be used as tree[targetedkid_ind]
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
	
	return res # test if True or False

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
	for i,w in enumerate(sentence):
		if w == node:
			node_id = i
			prev = sentence[node_id-1]
	
	return prev

###########################################################################
###########################################################################
## file-level counts for normalisation

def wordcount(trees):
	words = 0
	for tree in trees:
		words += len(tree)
	
	return words

## this function adjusts counts for number of sentences, accounting for ca. 4% of errors in EN/GE
# where sentence ends with colon or semi-colon; in RU this error makes only 0.4%
def sents_num(trees, lang):
	sentnum = 0
	if lang == 'en':
		for tree in trees:
			lastwd = tree[-1]
			if not lastwd[2] in [':', ';', 'Mr.', 'Dr.']:
				sentnum += 1
	if lang == 'de':
		for tree in trees:
			lastwd = tree[-1]
			if not lastwd[2] in [':', ';', 'z.B.', 'Dr.']:
				sentnum += 1
	if lang == 'ru':
		for tree in trees:
			lastwd = tree[-1]
			if not lastwd[2] in [':', ';', 'Дж.']:
				sentnum += 1  # this is a fair num-of-sents count for a file
	
	return sentnum

def verbs_num(trees, lang):
	verbs = 0
	for tree in trees:
		for w in tree:
			if w[3] == 'VERB':
				verbs += 1
	return verbs

###############################################################
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
		print(':'.join(i for i in [tu[0],str(tu[1])]), end ="; ")
	print("Dict size", len(list(dic_sort.items())))
	
	return tot

import os

def support_all_lang(lang):
    # Basisverzeichnis für die Listen
    lists_path = r"C:\Users\Fox0197\Desktop\projekt_final\searchlists"
    
    # Mappings für Sprachspezifische Dateien
    file_mappings = {
        'en': [
            "en_deverbals_stop.lst",
            "en_adv_quantifiers.lst",
            "en_modal-adj_predicates.lst",
            "en_converts.lst",
        ],
        'de': [
            "de_deverbals_stop.lst",
            "de_adv_quantifiers.lst",
            "de_modal-adj_predicates.lst",
            "de_converts.lst",
        ]
    }
    
    # Sicherstellen, dass die Sprache unterstützt wird
    if lang not in file_mappings:
        raise ValueError(f"Sprache '{lang}' wird nicht unterstützt.")
    
    # Listen vorbereiten
    pseudo_deverbs = []
    quantifiers = []
    adj_pred = []
    converts = []
    
    # Sprachspezifische Dateien laden
    file_names = file_mappings[lang]
    
    # Funktion für Datei-Inhalte
    def read_file(file_name):
        file_path = os.path.join(lists_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Die Datei {file_path} wurde nicht gefunden.")
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    
    # Dateien verarbeiten
    pseudo_deverbs = read_file(file_names[0])
    quantifiers = read_file(file_names[1])
    adj_pred = read_file(file_names[2])
    converts = read_file(file_names[3])
    
    return pseudo_deverbs, quantifiers, adj_pred, converts
	

def dms_support_all_langs(l):
    # Sicherstellen, dass lists_path definiert ist
    lists_path = '/Users/Fox0197/Desktop/translationese45/searchlists/'
    
    # Erstelle die Pfade basierend auf dem Parameter 'l'
    add = lists_path + "dms/" + l + "_additive.lst"
    add_list = [i.strip() for i in open(add, 'r', encoding='utf-8').readlines()]

    adv = lists_path + "dms/" + l + "_adversative.lst"
    adv_list = [i.strip() for i in open(adv, 'r', encoding='utf-8').readlines()]

    caus = lists_path + "dms/" + l + "_causal.lst"
    caus_list = [i.strip() for i in open(caus, 'r', encoding='utf-8').readlines()]

    temp_sequen = lists_path + "dms/" + l + "_temp_sequen.lst"
    temp_sequen_list = [i.strip() for i in open(temp_sequen, 'r', encoding='utf-8').readlines()]

    epistem = lists_path + "dms/" + l + "_epistemic.lst"
    epist_list = [i.strip() for i in open(epistem, 'r', encoding='utf-8').readlines()]

    return add_list, adv_list, caus_list, temp_sequen_list, epist_list