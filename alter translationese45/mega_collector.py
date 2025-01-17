'''
produces a dictionary with each text values for each feature + a class value (type) for three languages
the language index is taken from the folder name and needs to be set up accordingly
'''
import os, sys
import csv
from extractors import *

rootdir = 'C:/Users/Fox0197/Desktop/projekt_final//'
data = rootdir + 'pivot'
outname = rootdir + 'testversion.tsv'

# here, for each file we collect counts averaged over number of words or number of sentences
## muted features: passives interrog andor wdlength mark nn
keys = 'afile alang akorp astatus ' \
       'sentlength ppron possdet indef cconj whconj relativ pied correl copula ' \
       'attrib pasttense lexdens lexTTR mquantif mpred finites infs pverbals deverbals ' \
       'bypassives longpassives sconj addit advers caus tempseq epist but comp ' \
       'sup neg numcls simple demdets nnargs mhd mdd acl aux ' \
       'aux:pass ccomp nsubj:pass parataxis xcomp'.split()
master_dict = {k: [] for k in keys}

basic_stats = {}
languages = ['en', 'de', 'ru']
adv_support = {}
mpred_support = {}
pseudo_deverbs = {}
vconverts = {}

additive = {}
adversative = {}
causal = {}
sequen = {}
epistem = {}

for l in languages:
	# import all the lists for the three languages and add them to a lang-dictionary with three lang-keys
	adv_lst, mpred_lst, pseudo_deverbs_lst, vconverts_lst = support_all_lang(l)
	adv_support[l] = adv_lst
	mpred_support[l] = mpred_lst
	pseudo_deverbs[l] = pseudo_deverbs_lst
	vconverts[l] = vconverts_lst
	
	additive_lst, adversative_lst, causal_lst, sequen_lst, epistem_lst = dms_support_all_langs(l)
	additive[l] = additive_lst
	adversative[l] = adversative_lst
	causal[l] = causal_lst
	sequen[l] = sequen_lst
	epistem[l] = epistem_lst
	
	print('---', file=sys.stderr)
	print('Importing support lists for %s:' % l.upper(), file=sys.stderr)
	print('==%s adverbial qualtifiers' % len(adv_support[l]), file=sys.stderr)
	print('==%s modal predicative adjectives' % len(mpred_support[l]),file=sys.stderr)
	print('==%s stopwords for deverbal nouns' % len(pseudo_deverbs[l]), file=sys.stderr)
	print('==%s verbal nouns by conversion' % len(vconverts[l]), file=sys.stderr)
	print('',file=sys.stderr)
	print('Importing DM searchlists for %s:' % l.upper(), file=sys.stderr)
	print('==%s additive' % len(additive[l]), file=sys.stderr)
	print('==%s adversative' % len(adversative[l]),file=sys.stderr)
	print('==%s causative' % len(causal[l]), file=sys.stderr)
	print('==%s temporal/sequencial' % len(sequen[l]), file=sys.stderr)
	print('==%s DM of epistemic stance' % len(epistem[l]), file=sys.stderr)
	
for subdir, dirs, files in os.walk(data):
	for i, file in enumerate(files):
		
		filepath = subdir + os.sep + file
		last_folder = subdir + os.sep
		
		# Extrahiere das Sprachkürzel aus dem Verzeichnisnamen
		language = os.path.basename(os.path.normpath(last_folder))

		# Sicherstellen, dass language nur die erwarteten Kürzel enthält
		if language not in languages:raise KeyError(f"Ungültiges Sprachkürzel '{language}'. Erwartet: {languages}. Überprüfe die Ordnerstruktur.")		

		# data hierachy: /your/path/preprocessed/croco/pro/de/*.conllu
		
		# this collects counts for every sentence in a document
		# prepare for writing metadata:
		lang, korp, status = get_meta(last_folder)
		meta_str = '_'.join(get_meta(last_folder)) # lang, korp, status
		
		if i % 20 == 0:
			print('I have processed %s files from %s' % (i, meta_str.upper()), file=sys.stderr)
		
		# don't forget the filename
		doc = os.path.splitext(os.path.basename(last_folder + filepath))[0]  # without extention

		data = open(filepath, encoding='utf-8').readlines()
		
		corp_id = lang + '_' + status + '_' + korp
		sents = get_trees(data)
		if corp_id in basic_stats.keys():
			basic_stats[corp_id] += len(sents)
		else:
			basic_stats[corp_id] = len(sents)
		
		current = {}
		
		# call functions that operate at doc-level and write to dic for current file
		# get text parameters for normalization
		normBy_wc = wordcount(sents)
		normBy_sentnum = sents_num(sents, language)
		normBy_verbnum = verbs_num(sents, language)
		
		## create text-level counters for each feature values
		ppron_res = 0
		possdet_res = 0
		anysome_res = 0
		cconj_res = 0
		sconj_res = 0
		advconj_res = 0
		relativ_res = 0
		pied_res = 0
		correl_res = 0
		copula_res = 0
		# wdlength_res = 0
		# interrog_res = 0
		# nn_res = 0
		attrib_res = 0
		pasttense_res = 0
		lex_ty_res = 0
		lex_to_res = 0
		mpred_res = 0
		mquantif_res = 0
		finites_res = 0
		### adding July2019 features
		infs_res = 0
		pverbals_res = 0
		# passives_res = 0
		bypassives_res = 0
		longpassives_res = 0
		
		speakdiff_res = 0
		readerdiff_res = 0
		
		## text-level counts
		deverbals_res = nominals(sents, language, pseudo_deverbs, vconverts)
		
		addit_res = count_dms(additive, sents, language)
		advers_res = count_dms(adversative, sents, language)
		caus_res = count_dms(causal, sents, language)
		tempseq_res = count_dms(sequen, sents, language)
		epist_res = count_dms(epistem, sents, language)+get_epistemic_stance(sents, language)
		
		# andor_res = and_or_counts(sents, language)
		but_res = but_counts(sents, language)
		
		## counts for degrees of comparison normalized internally for num of adj+adv
		comp_res, sup_res = comparison_degrees(sents, language)
		
		neg_res = polarity(sents, language)
		
		# average number of clauses per sentence and ratio of simple sentences in text
		numcls_res, simple_res = sents_complexity(sents)
		
		demdets_res = demdeterm(sents, language)
		nnargs_res = nouns_to_all(sents)
		
		## run functions and get absolute freqs for each text
		for sent in sents:
			ppron_res += prsp(sent, language)[0]
			possdet_res += possdet(sent, language)[0]
			anysome_res += anysome(sent, language)[0]
			cconj_res += cconj(sent, language)[0]
			sconj_res += sconj(sent, language)[0]
			advconj_res += whconj(sent, language)[0]
			mhd = speakdiff(sent)
			if mhd:
				speakdiff_res += speakdiff(sent)
				readerdiff_res += readerdiff(sent)
			rel,_,ppiping,correlat = relativ(sent, language)
			relativ_res += rel
			pied_res += ppiping
			correl_res += correlat
			copula_res += copulas(sent)
			# wdlength_res += word_length(sent)
			# interrog_res += interrog(sent, language)[0]
			# nn_res += nn(sent, language)[0]
			attrib_res += attrib(sent)[0]
			pasttense_res += pasttense(sent)
			ty, to = lex_ty_to(sent, language) # counts of content types and tokens are needed elsewhere
			lex_ty_res += ty
			lex_to_res += to
			mpred_res += modpred(sent, language,mpred_support)[0]
			mquantif_res += advquantif(sent, language,adv_support)[0]
			finites_res += finites(sent,language)[0]
			infs_res += infinitives(sent,language,mpred_support)
			pverbals_res += participles(sent,language)
			bys, nobys = passives(sent, language)
			# passives_res += (bys+nobys)
			bypassives_res += bys
			longpassives_res += nobys
			# add functions here, except for these that operate on the text, rather than sentence level
		
		# run functions that are doc(file)-level
		avsents = av_s_length(sents, language)
		# and add the values to the dic for this text
		current['sentlength'] = avsents
		
		# normalisation for the absolute sentence-level freqs is done mostly(NB!) in two different ways:
		## by number of words in the text
		current['ppron'] = ppron_res/normBy_wc
		current['possdet'] = possdet_res / normBy_wc
		current['indef'] = anysome_res / normBy_wc
		current['cconj'] = cconj_res / normBy_sentnum
		current['sconj'] = sconj_res / normBy_sentnum
		current['whconj'] = advconj_res / normBy_sentnum
		# current['nn'] = nn_res / normBy_wc
		current['lexdens'] = lex_ty_res / normBy_wc
		current['mquantif'] = mquantif_res / normBy_wc
		
		current['lexTTR'] = lex_ty_res / lex_to_res
		
		current['mpred'] = mpred_res / normBy_sentnum # need to normalize to number of finites when I get it :-)
		
		## by number of sentences
		current['mhd'] = speakdiff_res / normBy_sentnum
		current['mdd'] = readerdiff_res / normBy_sentnum
		current['relativ'] = relativ_res / normBy_sentnum
		current['pied'] = pied_res / normBy_sentnum
		current['correl'] = correl_res / normBy_sentnum
		current['copula'] = copula_res / normBy_sentnum
		# current['interrog'] = interrog_res / normBy_sentnum
		current['attrib'] = attrib_res / normBy_sentnum
		current['pasttense'] = pasttense_res / normBy_sentnum
		# current['wdlength'] = wdlength_res / normBy_sentnum
		current['finites'] = finites_res / normBy_verbnum
		
		## alternatively normalize these three features by number of verbs using / normBy_verbnum
		current['infs'] = infs_res / normBy_verbnum
		current['pverbals'] = pverbals_res / normBy_verbnum
		current['deverbals'] = deverbals_res / normBy_verbnum
		
		# current['passives'] = passives_res / normBy_sentnum
		current['bypassives'] = bypassives_res / normBy_sentnum ## maybe these two need to be counted as one feature
		current['longpassives'] = longpassives_res / normBy_sentnum
		## here are counts for 5 semantic groups of DMs + conts for and/or and but (maybe merge them!)
		current['addit'] = addit_res / normBy_sentnum
		current['advers'] = advers_res / normBy_sentnum
		current['caus'] = caus_res / normBy_sentnum
		current['tempseq'] = tempseq_res / normBy_sentnum
		current['epist'] = epist_res / normBy_sentnum
		# current['andor'] = andor_res / normBy_sentnum
		current['but'] = but_res / normBy_sentnum
		current['comp'] = comp_res
		current['sup'] = sup_res
		current['neg'] = neg_res / normBy_sentnum
		current['numcls'] = numcls_res
		current['simple'] = simple_res
		current['demdets'] = demdets_res / normBy_wc
		current['nnargs'] = nnargs_res
		
		## add 7 UD features from a dict
		dep_prob_dict = ud_probabilities(sents, language)
		for k, val in dep_prob_dict.items():
			current[k] = val
		
		# get filename, text type (learner, pro, ref) and register to the features
		current['afile'] = doc
		current['alang'] = lang
		current['akorp'] = korp
		current['astatus'] = status
		
		# re-writing the dictionary to get the frequencies from all 10 subcorpora in one database to be written as a tsv spreadsheet
		for key in master_dict.keys():
			master_dict[key].append(current[key])

	with open(outname, "w") as outfile:
	
		writer = csv.writer(outfile, delimiter="\t")
		writer.writerow(keys)

		writer.writerows(zip(*[master_dict[key] for key in keys]))

print('Your data is ready. Lets see whether we can see any patterns in it')