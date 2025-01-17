"""
from conllu format, this script produces a table with documents/filenames in rows and features in columns + meta columns
- the language index is taken from the folder name and needs to be set up accordingly
- note the expected structure of folders: we store names of the last several folders as values for metadata columns
(1) the last folder is the languages (en, ru) these indices need to be passed to the --langs option
(2) the last but one is status (professional/student, reference, source
(in Kateryna's project it duplicates the lang folder for the sake of structure)
(3) the last but two is source of data (authors, corpora, genres/registers)
For example:
data hierarchy: /your/path/anylength/parsed/register/status/lang/*.conllu, where parsed is the name of the input folder

USAGE (from kateryna/get_feats/extraction/ folder!):
python3 get_feats/extraction/new_mega_collector.py --input corpus/parsed/ --output data/debates_fiction.tsv --levels doc register type lang
"""

import os
import csv
from extractors import av_s_length, word_length, interrog, nn, speakdiff, readerdiff, content_ty_to, finites, \
    attrib, pasttense, count_dms, get_epistemic_stance, sents_complexity, ud_probabilities, ud_freqs, nouns_to_all
from extractors import prsp, possdet, anysome, cconj, sconj, copulas, polarity, demdeterm, propn, preps
from helpfunctions import dms_support_all_langs, get_trees, wordcount, sents_num, verbs_num
from collections import defaultdict
import time

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='corpus/parsed/',
                        help="Path to the tree of folders with *.conllu named after your text categories",
                        required=True)
    parser.add_argument('--output', default='data/debates_fiction.tsv', help="Path to, and name of, the resulting spreadsheet",
                        required=True)
    parser.add_argument('--supports', default='get_feats/extraction/searchlists/', help="Path to the folder with lists")
    parser.add_argument('--minlen', default=2, type=int, help="Minimum allowed sent length")
    parser.add_argument('--langs', nargs='+', default=['en', 'es'], help='Pass language indices like so: --langs en es')
    parser.add_argument('--levels', nargs='+', default=['doc', 'register', 'type', 'lang'],
                        help='Levels under rootdir. Example: for clean/ted/ref/ru/ --levels doc register type lang')
    start = time.time()
    args = parser.parse_args()

    input_dir = args.input  # corpus/parsed/

    os.makedirs('data/', exist_ok=True)
    outname = args.output  # data/debates.tsv

    # here, for each file we collect counts averaged over number of words or number of sentences

    meta = args.levels

    # 29 extractors
    ud_features = 'sentlength wdlength interrog nn mhd mdd content_dens content_TTR finites attrib ' \
                  'pasttense addit advers caus tempseq epist numcls simple nnargs ppron ' \
                  'possp intonep cconj sconj neg copula determ propn adp'.split()  # these are newly added
    # UD relations
    # udrels = "acl aux aux:pass ccomp nsubj:pass parataxis xcomp".split()  # this is the selection effective for EN>RU
    # 31 items: we are not using variants of rels, except we use 'aux:pass' instead of 'aux', and we drop 'cop',
    # 'conj', 'csubj', 'root', 'det', 'punct' because they are duplicated by custom extractors
    all_udrels = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux:pass', 'case', 'cc', 'ccomp', 'clf',
                  'compound', 'dep', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list',
                  'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'reparandum', 'vocative',
                  'xcomp']

    # 17: we rely on some of these tags and predefined lists to filter out annotation errors;
    # in some cases we employ grammatical categories (ex. 'PronType=Tot') for extracting finer-defined categories
    # all_upos = ['PROPN', 'PRON', 'PART', 'PUNCT', 'ADP', 'NUM', 'INTJ', 'AUX', 'ADV', 'ADJ', 'CCONJ', 'X', 'SCONJ',
    #             'DET', 'SYM', 'NOUN', 'VERB']

    keys = meta + ['wc', 'sents'] + ud_features + all_udrels

    master_dict = {k: [] for k in keys}

    basic_stats = defaultdict(int)
    languages = args.langs

    additive = {}
    adversative = {}
    causal = {}
    sequen = {}
    epistem = {}

    for lang in languages:
        addit, advers, caus, seque, epist = dms_support_all_langs(lang, lists_path=args.supports)
        additive[lang] = addit
        adversative[lang] = advers
        causal[lang] = caus
        sequen[lang] = seque
        epistem[lang] = epist

        print('---')
        print('Importing DM searchlists for %s:' % lang.upper())
        print('==%s additive' % len(additive[lang]))
        print('==%s adversative' % len(adversative[lang]))
        print('==%s causative' % len(causal[lang]))
        print('==%s temporal/sequencial' % len(sequen[lang]))
        print('==%s DM of epistemic stance' % len(epistem[lang]))

    tot_bads = 0
    tot_shorts = 0
    counter = 0
    for subdir, dirs, files in os.walk(input_dir):
        for i, file in enumerate(files):
            filepath = subdir + os.sep + file
            last_folder = subdir + os.sep

            path_to_last_folder = subdir
            # levels has values for meta[1:] keys in the master and current dicts
            levels = path_to_last_folder.split(os.sep)[-(len(args.levels) - 1):]
            corp_id = '_'.join(levels)

            language = levels[-1]

            # don't forget the filename without extention
            doc = file.rstrip('.conllu')
            data = open(filepath, encoding="utf-8").readlines()

            sents, bads, shorts = get_trees(data, minlen=args.minlen)
            tot_bads += bads
            tot_shorts += shorts

            if i % 50 == 0:
                print(f'I have processed {i} files from {corp_id.upper()}')
                print(f'{tot_bads} all-punct-num sents and additionally {tot_shorts} less-than-{args.minlen}-meaningful-words sents skipped')
                print()

            basic_stats[corp_id] += len(sents)

            # initialising a dict for the current document with doc:filename key-value pair
            current = {args.levels[0]: doc}

            # call functions that operate at doc-level and write to dic for current file
            # get text parameters for normalization
            normBy_wc = wordcount(sents)
            normBy_sentnum = sents_num(sents, language)
            normBy_verbnum = verbs_num(sents)

            # create doc-level counters for each feature whose values are collected at sentence level

            wc = normBy_wc
            sent_count = normBy_sentnum

            wdlength_res = 0
            interrog_res = 0
            nn_res = 0
            speakdiff_res = 0
            readerdiff_res = 0
            content_ty_res = 0
            content_to_res = 0
            finites_res = 0
            attrib_res = 0
            pasttense_res = 0

            # these require knowledge of Spanish
            ppron_res = 0
            possdet_res = 0
            anysome_res = 0
            cconj_res = 0
            sconj_res = 0
            demdets_res = 0
            copula_res = 0
            propn_res = 0
            adp_res = 0
            neg_res = 0

            # features that collect counts for doc-level (no counters required)
            avsents = av_s_length(sents, language)
            addit_res = count_dms(additive, sents, language)
            advers_res = count_dms(adversative, sents, language)
            caus_res = count_dms(causal, sents, language)
            tempseq_res = count_dms(sequen, sents, language)
            epist_res = count_dms(epistem, sents, language) + get_epistemic_stance(sents, language)
            # average number of clauses per sentence and ratio of simple sentences in text
            numcls_res, simple_res = sents_complexity(sents)
            nnargs_res = nouns_to_all(sents)

            # run functions and collect freqs for each text
            for sent in sents:
                mhd = speakdiff(sent)
                if mhd:
                    speakdiff_res += speakdiff(sent)
                    readerdiff_res += readerdiff(sent)

                wdlength_res += word_length(sent)
                interrog_res += interrog(sent)[0]
                nn_res += nn(sent)[0]
                attrib_res += attrib(sent)[0]
                pasttense_res += pasttense(sent)
                ty, to = content_ty_to(sent)  # counts of content types and tokens are needed elsewhere
                content_ty_res += ty
                content_to_res += to
                finites_res += finites(sent)

                ppron_res += prsp(sent, language)[0]
                possdet_res += possdet(sent, language)[0]
                anysome_res += anysome(sent, language)[0]
                cconj_res += cconj(sent, language)[0]
                sconj_res += sconj(sent, language)[0]
                demdets_res += demdeterm(sent, language)
                copula_res += copulas(sent)
                neg_res += polarity(sent, language)
                propn_res += propn(sent)
                adp_res += preps(sent, language)

            # indep_features = 'sentlength wdlength interrog nn mhd mdd lexdens lexTTR finites attrib pasttense ' \
            #                  'addit advers caus tempseq epist numcls simple nnargs'.split()

            # add the values collected at doc- or sent-level to the dic
            current['wc'] = wc
            current['sents'] = sent_count
            current['sentlength'] = avsents
            current['wdlength'] = wdlength_res / normBy_sentnum
            # normalisation
            current['interrog'] = interrog_res / normBy_sentnum
            current['nn'] = nn_res / normBy_wc
            current['mhd'] = speakdiff_res / normBy_sentnum
            current['mdd'] = readerdiff_res / normBy_sentnum
            current['content_dens'] = content_ty_res / normBy_wc
            current['content_TTR'] = content_ty_res / content_to_res
            current['finites'] = finites_res / normBy_verbnum
            current['attrib'] = attrib_res / normBy_sentnum
            current['pasttense'] = pasttense_res / normBy_sentnum
            # here are counts for 5 semantic groups of DMs + conts for and/or and but (maybe merge them!)
            current['addit'] = addit_res / normBy_sentnum
            current['advers'] = advers_res / normBy_sentnum
            current['caus'] = caus_res / normBy_sentnum
            current['tempseq'] = tempseq_res / normBy_sentnum
            current['epist'] = epist_res / normBy_sentnum
            current['numcls'] = numcls_res
            current['simple'] = simple_res
            current['nnargs'] = nnargs_res

            current['ppron'] = ppron_res / normBy_wc
            current['possp'] = possdet_res / normBy_wc
            current['intonep'] = anysome_res / normBy_wc
            current['cconj'] = cconj_res / normBy_sentnum
            current['sconj'] = sconj_res / normBy_sentnum
            current['copula'] = copula_res / normBy_sentnum
            current['neg'] = neg_res / normBy_sentnum
            current['determ'] = demdets_res / normBy_wc
            current['propn'] = propn_res / normBy_sentnum
            current['adp'] = adp_res / normBy_wc

            # add UD features

            # if you want to use UD probabilities (normalisation to wc on sent-level)
            # dep_dict = ud_probabilities(sents, udfeats_=all_udrels)
            # for k, val in dep_dict.items():
            #     current[k] = val

            # if you want to use freqs noemalised to sentence counts at doc-level
            dep_dict = ud_freqs(sents, udfeats_=all_udrels)
            for k, val in dep_dict.items():
                current[k] = val / normBy_sentnum

            # get filename, text type (src, tgt, ref) and register to the features
            for var, val in zip(args.levels[1:], levels):
                current[var] = val

            # re-writing the dictionary to get the frequencies from all subcorpora into one spreadsheet
            for key in master_dict.keys():
                master_dict[key].append(current[key])
            counter += 1
    with open(outname, "w") as outfile:

        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(keys)

        writer.writerows(zip(*[master_dict[key] for key in keys]))

    print(f'Your data {len(master_dict["doc"])} is ready. Lets see whether we can see any patterns in it')
    print(f'We used {len(ud_features)} custom-made features and {len(all_udrels)} default tags')

    end = time.time()
    processing_time = int(end - start)
    print(f'Feature extraction from {counter} files took {(processing_time / 60):.2f} minutes')
