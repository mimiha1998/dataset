"""
this script prints examples (sentences) that are matched by the functions provided
"""

import os
import csv
from helpfunctions import dms_support_all_langs, get_trees, wordcount, sents_num, verbs_num
from collections import defaultdict
import argparse
from collections import Counter


# include only noun substituters, i.e. pronouns par excellence, of Indefinite, total and negative semantic subtypes
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


def sconj(tree, lang):
    count = 0
    matches = []
    for w in tree:
        if lang == 'en':
            if 'SCONJ' in w[3] and w[2] in ['that', 'how', 'if', 'after', 'before', 'when', 'as', 'while', 'because',
                                            'for', 'whether', 'although', 'though', 'since', 'once', 'so', 'until',
                                            'despite', 'unless', 'whereas', 'whilst']:
            # if 'SCONJ' in w[3] and w[2] in ['that', 'if', 'as', 'of', 'while', 'because', 'by', 'for', 'to', 'than',
            #                                 'whether', 'in', 'about', 'before', 'after', 'on', 'with', 'from', 'like',
            #                                 'although', 'though', 'since', 'once', 'so', 'at', 'without', 'until',
            #                                 'into', 'despite', 'unless', 'whereas', 'over', 'upon', 'whilst', 'beyond',
            #                                 'towards', 'toward', 'but', 'except', 'cause', 'together']:
                count += 1
                matches.append(w[2].lower())

        elif lang == 'es':
            if 'SCONJ' in w[3] and w[2] in "así aun aunque como conque cuando donde luego por porque pues que salvo si".split():
                count += 1
                matches.append(w[2].lower())

    return count, matches


def copulas(tree):
    cases1 = []
    cases2 = []
    copCount = 0
    for i, w in enumerate(tree):
        if w[7] == "cop" and w[2] in ['be', 'ser', 'estar']:

            for prev_w in [tree[i - 1], tree[i - 2], tree[i - 3]]:
                if prev_w[2] == 'there':
                    copCount += -1
                    continue
            if w[2] == 'estar':
                cases1.append(' '.join(w[1] for w in tree))
            copCount += 1

        if w[2] == "estar":
            copCount += 1
            cases2.append(' '.join(w[1] for w in tree))

    return copCount, cases1, cases2


def preps(tree, lang):
    res = 0
    cases = []
    for w in tree:
        if lang == 'en':
            if w[3] == 'ADP':  # and w[2] in []:
                res += 1
                cases.append(w[2])
        elif lang == 'es':
            if w[3] == 'ADP' and w[2] in "a ante bajo con contra de desde durante en entre hacia hasta mediante " \
                                         "para por según sin sobre tras excepto salvo incluso".split():
                res += 1
                cases.append(w[2])
    return res, cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='corpus/parsed/debates/src/en/',
                        help="Path to the tree of folders with *.conllu named after your text categories")
    parser.add_argument('--minlen', default=2, type=int, help="Minimum allowed sent length")
    parser.add_argument('--lang', default='en', help='Pass a language index')

    args = parser.parse_args()
    input_dir = args.input
    language = args.lang

    tot_bads = 0
    tot_shorts = 0
    allofthem1 = []
    allofthem2 = []
    all_deprels = []
    for subdir, dirs, files in os.walk(input_dir):
        for i, file in enumerate(files):
            filepath = subdir + os.sep + file
            data = open(filepath).readlines()

            sents, bads, shorts = get_trees(data, minlen=args.minlen)
            tot_bads += bads
            tot_shorts += shorts

            for sent in sents:
                res, ex1 = preps(sent, language)
                # res, ex1, ex2 = copulas(sent)
                # res, ex = sconj(sent, language)
                # for w in sent:
                #     all_deprels.append(w[7].strip())
                if ex1:
                    allofthem1.extend(ex1)
                # if ex2:
                #     allofthem2.extend(ex2)

    print(f'Which items in this set were not asked for? \n {set(allofthem1)}')
    freq_dict = Counter(allofthem1)
    good2go = []
    for k, v in freq_dict.items():
        if v > 10:
            print(k, v)
            good2go.append(k)
    print(good2go)
    # for idx, i in enumerate(allofthem1):
    #     print(idx, i)
    # print(f'Total number of deprels established empirically is {len(set(all_deprels))}: {set(all_deprels)}')
    # print(f'{tot_bads} all-punct-num sents and additionally {tot_shorts} less-than-{args.minlen}-meaningful-words sents skipped')
    print('\nDONE')
