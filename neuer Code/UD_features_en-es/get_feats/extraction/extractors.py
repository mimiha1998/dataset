"""
Updated 6 December 2021 from Feb 5-10,2019 version
- contains functions that are called from mega_collector.py to extract lang-independent features from conllu format;
- calls functions from helpfunctions.py to traverse conllu sentence trees
- each word is represented as: int(identifier), token, lemma, upos, xpos, feats, int(head), rel
- pip install igraph  (library by Tamas Nepusz)
"""

import numpy as np
from igraph import *
from igraph._igraph import arpack_options
from helpfunctions import has_kid_by_lemlist, has_auxkid_by_tok


# corrected sentence length for the file
def av_s_length(trees, lang):
    sent_lengths = []
    if lang == 'en':
        texts = trees.copy()  # the riddle solved! Make a real copy of the input variable!
        for i, tree in enumerate(texts):
            if i < len(texts):
                lastwd = tree[-1]
                if lastwd[1] in [':', ';', 'Mr.', 'Dr.']:
                    if len(texts) > i + 1:
                        nextsent = texts[i + 1]
                        sent_lengths.append(len(tree) + len(nextsent))
                        texts.remove(texts[i + 1])
                    else:
                        sent_lengths.append(len(tree))
                else:
                    sent_lengths.append(len(tree))
    elif lang == 'es':
        texts = trees.copy()
        for i, tree in enumerate(texts):
            if i < len(texts):
                lastwd = tree[-1]
                if lastwd[1] in [':', ';']:
                    try:
                        nextsent = texts[i + 1]
                        sent_lengths.append(len(tree) + len(nextsent))
                        texts.remove(texts[i + 1])
                    except IndexError:
                        sent_lengths.append(len(tree))
                        print('Despite filtering I still have texts that finish in no end-of-sent punct!')
                        print(' '.join(w[1] for w in tree))
                        continue
                else:
                    sent_lengths.append(len(tree))
    else:
        print('Specify the language, please', file=sys.stderr)  # the path to file doesn't satisfy the requirements

    return np.average(sent_lengths)


def word_length(tree):
    words = 0
    letters = 0
    for el in tree:
        if not el[1] in ['.', ',', '!', '?', ':', ';', '"', '-', '—', '(', ')']:
            words += 1
            letters += len(el[1])
        else:
            continue
    if words == 0:
        print(tree)
    av_wordlength = letters / words
    # print(letters, words)
    return av_wordlength


# this feature is fishy, because I deleted all sents that are less than 3 words; many of them are interrogative
def interrog(tree):
    count = 0
    matches = []
    last = tree[-1]
    lastbut = tree[-2]
    if tree[-1][2] == '?' or tree[-2][2] == '?':
        count += 1
        matches.append(last[2])
        matches.append(lastbut[2])

    return count, matches


def nn(tree):
    count = 0
    matches = []
    for w in tree:
        lemma = w[2].lower()
        if 'NOUN' in w[3]:
            count += 1
            matches.append(lemma)

    return count, matches


# graph-based feature

# this function tests connectedness of the graph; it helps to skip malformed sentences
# additionally it produces the sentence graph
# (NB!! it slows down the process, hence if graphs are needed call a separate function)
def test_sanity(tree):
    arpack_options.maxiter = 3000
    bad_trees = 0
    sentence_graph = Graph(len(tree) + 1)

    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[1] for word in tree]  # string of vertices' attributes called name
    # the name attribute is renamed label,
    # because in drawing vertex, labels are taken from the label attribute by default
    sentence_graph.vs["label"] = sentence_graph.vs["name"]
    edges = [(word[6], word[0]) for word in tree if word[7] != 'punct']  # (int(identifier), int(head), token, rel)
    try:
        sentence_graph.add_edges(edges)
        sentence_graph.vs.find("ROOT")["shape"] = 'diamond'
        sentence_graph.vs.find("ROOT")["size"] = 40
        disconnected = [vertex.index for vertex in sentence_graph.vs if vertex.degree() == 0]
        sentence_graph.delete_vertices(disconnected)
    except:
        bad_trees += 0

    return sentence_graph


def speakdiff(tree):
    # call the above function to get the graph and counts for disintegrated trees
    # (syntactic analysis errors of assigning dependents to punctuation)
    graph = test_sanity(tree)

    parts = graph.components(mode=WEAK)
    mhd = 0
    errors = 0
    if len(parts) == 1:
        nodes = [word[1] for word in tree if word[7] != 'punct' and word[7] != 'root']
        all_hds = []  # or a counter = 0?
        for node in nodes:
            try:
                hd = graph.shortest_paths_dijkstra('ROOT', node, mode=ALL)
                all_hds.append(hd[0][0])
            except ValueError:
                errors += 1
        # print(tree)
        if all_hds:
            mhd = np.average(all_hds)
    # print(errors)
    return mhd


# this is a slower function which allows to print graphs for sentences (to be used in overall_freqs)
def speakdiff_visuals(tree):
    # call the above function to get the graph and counts for disintegrated trees
    # (syntactic analysis errors of assigning dependents to punctuation)
    graph = test_sanity(tree)
    # this is needed to see disconnected graphs from overall_freqs file
    try:
        communities = graph.community_leading_eigenvector()
    except InternalError:
        communities = ['dummy']
    av_degree = np.average(graph.degree(type="out"))
    parts = graph.components(mode=WEAK)
    mhd = 0
    if len(parts) == 1:
        nodes = [word[1] for word in tree if word[7] != 'punct' and word[7] != 'root']
        all_hds = []  # or a counter = 0?
        for node in nodes:
            hd = graph.shortest_paths_dijkstra('ROOT', node, mode=ALL)
            all_hds.append(hd[0][0])
        if all_hds:
            mhd = np.average(all_hds)

    return mhd, av_degree, communities, graph


def readerdiff(tree):
    # calculate comprehension difficulty=mean dependency distance(MDD) as “the distance between words and their parents,
    # measured in terms of intervening words.” (Hudson 1995 : 16)

    s = [q for q in tree if q[7] != 'punct']

    inbtw = []
    if len(s) > 1:
        for s_word_id in range(len(s)):  # use s-index to refer to words
            w = s[s_word_id]
            head_id = w[6]
            if head_id == 0:
                continue
            s_head_id = None
            for s_word_id_2 in range(len(s)):
                w1 = s[s_word_id_2]
                if head_id == w1[0]:
                    s_head_id = s_word_id_2
                    break
            # print(file, type(s_head_id), type(s_word_id), no)
            dd = abs(s_word_id - s_head_id) - 1
            inbtw.append(dd)
    # use this function instead of overt division of list sum by list length: if smth is wrong you'll get a warning!
    mdd = np.average(inbtw)

    return mdd  # average MDD for the sentence


# exclude PUNCT, SYM, X from tokens
def content_ty_to(tree):
    content_types = []
    content_tokens = []
    for w in tree:
        if 'ADJ' in w[3] or 'ADV' in w[3] or 'VERB' in w[3] or 'NOUN' in w[3]:
            content_type = w[2] + '_' + w[3]
            content_types.append(content_type)
            content_token = w[1] + '_' + w[3]
            content_tokens.append(content_token)

    return len(set(content_types)), len(content_tokens)


def finites(tree):
    fins = 0
    for w in tree:
        if 'VerbForm=Fin' in w[5]:
            fins += 1

    return fins


# не учитывает координацию: Они верны нашей сильной, веселой и злой планете.
def attrib(tree):
    count = 0
    matches = []
    for w in tree:
        if ('ADJ' in w[3] or 'VerbForm=Part' in w[5]) and 'amod' in w[7]:
            count += 1
            matches.append(w[2])

    return count, matches


def pasttense(tree):
    count = 0
    for w in tree:
        if 'Tense=Past' in w[5]:
            count += 1

    return count


def count_dms(searchlists, trees, lang):
    res = 0
    lst = searchlists[lang]
    for tree in trees:
        sent = ' '.join(w[1] for w in tree)
        for i in lst:
            i = i.strip()
            try:
                if i[0].isupper():
                    padded0 = i + ' '
                    if padded0 in sent:
                        res += 1
                else:
                    padded1 = ' ' + i + ' '

                    # if not surrounded by spaces gets: the best areAS FOR expats for as for
                    # ex: enTHUSiastic for thus

                    if padded1 in sent or i.capitalize() + ' ' in sent:
                        # capitalization changes the count for additive from 1822 to 2182
                        # (or to 2120 if padded with spaces)
                        # (furthermore: 1 to 10)
                        res += 1
            except IndexError:
                print('out of range index (blank line in list?: \n', i, file=sys.stderr)
                pass
    return res


def get_epistemic_stance(trees, lang):
    verbs = 0

    for tree in trees:
        if lang == 'en':
            # include stance verbs in Tense=Pres with i, we as nsubj into the counts
            sent = ' '.join(w[1] for w in tree)
            for w in tree:
                if w[1] in ['argue', 'doubt', 'assume', 'believe', 'find'] \
                        and has_kid_by_lemlist(w, tree, ['I', 'we']) \
                        and not has_auxkid_by_tok(w, tree, 'did'):
                    verbs += 1
                if w[1] == 'say' and has_auxkid_by_tok(w, tree, 'would') \
                        and has_kid_by_lemlist(w, tree, ['I', 'we']):
                    verbs += 1
                if w[1] in ['convinced', 'persuaded'] and has_kid_by_lemlist(w, tree, ['be']) \
                        and has_kid_by_lemlist(w, tree, ['I', 'we']):
                    verbs += 1
                if w[1] == 'feel' and has_kid_by_lemlist(w, tree, ['I', 'we']) \
                        and ('feel like' in sent or 'feel that' in sent):
                    verbs += 1
        # add rules here
        if lang == 'es':
            continue

    return verbs


def sents_complexity(trees):
    types = ['csubj', 'acl:relcl', 'advcl', 'acl', 'xcomp', 'parataxis']
    simples = 0
    clauses_counts = []
    for tree in trees:
        this_sent_cls = 0
        for w in tree:
            if w[7] in types:
                this_sent_cls += 1
        if this_sent_cls == 0:
            simples += 1
        clauses_counts.append(this_sent_cls)
    return np.average(clauses_counts), simples / len(trees)


# this is a help function to produce distributions of UD dependencies in a sentence
def relation_counts(tree, i_relations):
    sent_relations = [w[7] for w in tree]
    counts = {rel: sent_relations.count(rel) for rel in i_relations}

    return counts


def ud_freqs(trees, udfeats_=None):
    relations = udfeats_

    relations_d = {rel: [] for rel in relations}
    for tree in trees:
        rel_distribution = relation_counts(tree, relations_d)
        # this returns frequencies for the picked dependencies in this sent
        for rel in relations_d.keys():  # reusing the empty dict
            # this collects and average the probability stats over sents of the text to the global empty dict
            relations_d[rel].append(rel_distribution[rel])
            # values in this dict are lists of probabilities in each sentence; need to average them

    dict_out = {}
    for rel in relations_d.keys():
        dict_out[rel] = np.average(relations_d[rel])

    return dict_out


def relation_distribution(tree, i_relations):
    sent_relations = [w[7] for w in tree]
    distribution = {rel: sent_relations.count(rel) for rel in i_relations}
    # count method counts acl:relcl for acl, which is a desired behavior
    # Converting to probability distribution
    # 'probabilities' are basically ratio of the rel in question to all token rels in the sentence,
    # i.e., in effect, normalisation to wc
    total = sum(distribution.values())

    if total:
        # Only if there are from our limited list of UD relations in the sentence
        for key in distribution:
            distribution[key] /= total

    return distribution


# OLDER approach: previous research has shown that aux, ccomp, acl:relcl, mark, xcomp, parataxis and nsubj:pass; aux:pass
# are good indicators of translationese for EN > RU
def ud_probabilities(trees, udfeats_=None):
    relations = udfeats_

    relations_d = {rel: [] for rel in relations}
    for tree in trees:
        rel_distribution = relation_distribution(tree, relations_d)
        # this returns probabilities for the picked dependencies in this sent
        for rel in relations_d.keys():  # reusing the empty dict
            # this collects and average the probability stats over sents of the text to the global empty dict
            relations_d[rel].append(rel_distribution[rel])
            # values in this dict are lists of probabilities in each sentence; need to average them

    dict_out = {}
    for rel in relations_d.keys():
        dict_out[rel] = np.average(relations_d[rel])

    return dict_out


# ratio of NOUNS+proper names in these functions to the count of these functions
def nouns_to_all(trees):
    count = 0
    nouns = 0
    for tree in trees:
        for w in tree:
            if w[7] in ['nsubj', 'obj', 'iobj']:
                count += 1
                if w[3] == 'NOUN' or w[3] == 'PROPN':
                    nouns += 1
    res = nouns / count
    return res


# FEATURES that require lists and knowledge of the lang
def prsp(tree, lang):
    count = 0
    matches = []
    for w in tree:
        token = w[1].lower()
        if lang == 'en':
            if 'PRON' in w[3] and 'Person=' in w[5] and not 'Poss=Yes' in w[5]:
                if token in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']:
                    count += 1
                    matches.append(w[2].lower())
        elif lang == 'es':
            if 'PRON' in w[3] and 'Person=' in w[5] and not 'Poss=Yes' in w[5]:
                if token in "yo tú vos usted él ella nosotros nosotras ustedes vosotros vosotras ellos ellas me te lo nos os los la las se le les mí ti sí conmigo contigo consigo".split():
                    count += 1
                    matches.append(w[2].lower())
    return count, matches


def possdet(tree, lang):
    count = 0
    matches = []
    for w in tree:
        lemma = w[2].lower()
        # own and eigen are not included as they do not compare to свой, it seems
        if lang == 'en':
            if lemma in ['my', 'your', 'his', 'her', 'its', 'our', 'their']:
                if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[5]:
                    count += 1
                    matches.append(w[2].lower())
        elif lang == 'es':
            if lemma in "mi mis tu tus su sus nuestro nuestros nuestra nuestras vuestro vuestros vuestra vuestras".split():
                if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[5]:
                    count += 1
                    matches.append(w[2].lower())
    return count, matches


# include noun substituters, i.e. pronouns par excellence, of Indefinite, total and negative semantic subtypes
def anysome(tree, lang):
    count = 0
    matches = []
    for w in tree:
        if lang == 'en':
            if w[2] in ['anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything',
                        'nobody', 'none', 'nothing', 'somebody', 'someone', 'something',
                        'elsewhere', 'nowhere', 'everywhere', 'somewhere', 'anywhere']:
                count += 1
                matches.append(w[2].lower())
        elif lang == 'es':
            annotated_types = ['PronType=Ind', 'PronType=Tot', 'PronType=Int,Rel', 'PronType=Neg']
            if w[2] in "todo toda todos todas ambos ambas cada cada uno cada una alguno alguna algunos algunas " \
                       "algún ninguno ninguna ningunos ningunas ningún alguien algo nada nadie varios varias " \
                       "cualquiera cualesquiera cuánto cuánta cuántos cuántas tonto tanta tantos tantas tan mucho " \
                       "mucha muchos muchas muy poco poca pocos pocas bastante bastantes demasiado demasiada demasiados " \
                       "demasiadas más menos".split() and any(s in w[5] for s in annotated_types):
                count += 1
                matches.append(w[2].lower())
    return count, matches


# for now we use "all tokens receiving the tag (SCONJ, CCONJ) in the morphology annotation,
# disregarding correlatives and MWE; no semantic distinction (though it can be translationally important!)
def cconj(tree, lang):
    count = 0
    matches = []
    for w in tree:
        if lang == 'en':
            if 'CCONJ' in w[3] and w[2] in ['and', 'but', 'or', 'both', 'yet', 'either',
                                            '&', 'nor', 'plus', 'neither', 'ether']:
                count += 1
                matches.append(w[2].lower())
        elif lang == 'es':
            if 'CCONJ' in w[3] and w[2] in "y e ni o u ni pero sino más tanto como cuanto así como sea ya bien".split():
                count += 1
                matches.append(w[2].lower())
    return count, matches


def sconj(tree, lang):
    count = 0
    matches = []
    for w in tree:
        if lang == 'en':
            if 'SCONJ' in w[3] and w[2] in ['that', 'how', 'if', 'after', 'before', 'when', 'as', 'while', 'because',
                                            'for', 'whether', 'although', 'though', 'since', 'once', 'so', 'until',
                                            'despite', 'unless', 'whereas', 'whilst']:
                count += 1
                matches.append(w[2].lower())

        elif lang == 'es':
            if 'SCONJ' in w[3] and w[2] in "así aun aunque como conque cuando donde luego por porque pues que salvo si".split():
                count += 1
                matches.append(w[2].lower())

    return count, matches


def polarity(tree, lang):
    negs = 0
    for w in tree:
        if lang == 'en':
            if w[2] in ['no', 'not', 'neither']:
                negs += 1
                # The UK is n't some offshore tax paradise .
                # America no longer has a Greatest Generation .
                # But almost no major economy scores in the top 10
        if lang == 'es':
            if w[2] in ['no', 'ni']:
                negs += 1
    return negs


def copulas(tree):
    copcount = 0
    for i, w in enumerate(tree):
        if w[7] == "cop" and w[2] in ['be', 'ser', 'estar']:
            try:
                for prev_w in [tree[i - 1], tree[i - 2], tree[i - 3]]:
                    if prev_w[2] == 'there':
                        copcount += -1
            except IndexError:
                copcount += 1
            copcount += 1
    return copcount


def demdeterm(tree, lang):
    res = 0
    for w in tree:
        if lang == 'en':
            # frequency-ranked
            if w[7] == 'det' and w[2] in ['the', 'a', 'this', 'some', 'these', 'that', 'any', 'all',
                                          'every', 'another', 'each', 'those', 'either', 'such']:
                res += 1

        if lang == 'es':
            if w[7] == 'det' and w[2] in "el la los las lo al del un una unos unas este esta esto estos " \
                                         "estas ese esa eso esos esas aquel aquella aquello aquellos aquellas " \
                                         "tanto tanta tantos tantas tal tales tan".split():
                res += 1
    return res


def propn(tree):
    res = 0
    for w in tree:
        if w[3] == 'SCONJ':
            res += 1

    return res


def preps(tree, lang):
    res = 0
    for w in tree:
        if lang == 'en':
            if w[3] == 'ADP' and w[2] in ['of', 'in', 'unlike', 'for', 'at', 'as', 'to', 'along', 'with', 'after',
                                          'on', 'towards', 'amongst', 'within', 'over', 'during', 'by', 'against',
                                          'about', 'out', 'from', 'without', 'into', 'like', 'up', 'between',
                                          'before', 'down', 'across', 'per', 'off', 'around', 'since', 'onto',
                                          'through', 'beyond', 'under', 'despite', 'than', 'until', 'because',
                                          'upon', 'among', 'back', 'behind', 'past', 'outside', 'throughout',
                                          'inside', 'via', 'above', 'alongside', 'versus', 'below', 'round']:
                res += 1
        elif lang == 'es':
            if w[3] == 'ADP' and w[2] in "a ante bajo con contra de desde durante en entre hacia hasta mediante " \
                                         "para por según sin sobre tras excepto salvo incluso".split():
                res += 1
    return res
