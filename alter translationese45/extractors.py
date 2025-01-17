#! /usr/bin/python3
# coding: utf-8

# this script has functions that are called from mega_collector.py to extract linguistic features from texts in EN/GE/RU in conllu-format
# it calls help functions to traverse conllu sentence trees from helpfunctions.py
# each word is represented as: int(identifier), token, lemma, upos, xpos, feats, int(head), rel
# Feb 5-10,2019

import numpy as np
import re
from igraph import *
import itertools
from helpfunctions import *


## corrected sentence length for the file
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
	elif lang == 'de':
		texts = trees.copy()
		for i, tree in enumerate(texts):
			if i < len(texts):
				lastwd = tree[-1]
				if lastwd[1] in [':', ';', 'z.B.', 'Dr.']:
					nextsent = texts[i + 1]
					sent_lengths.append(len(tree) + len(nextsent))
					texts.remove(texts[i + 1])
				else:
					sent_lengths.append(len(tree))
	elif lang == 'ru':
		texts = trees.copy()
		for i, tree in enumerate(texts):
			if i < len(texts):
				lastwd = tree[-1]
				if lastwd[1] in [':', ';', 'Дж.']:
					nextsent = texts[i + 1]
					sent_lengths.append(len(tree) + len(nextsent))
					texts.remove(texts[i + 1])
				else:
					sent_lengths.append(len(tree))
	else:
		print('Specify the language, please', file=sys.stderr)  # the path to file doesn't satisfy the requirements
	
	return np.average(sent_lengths)


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
		elif lang == 'de':
			if 'PRON' in w[3] and 'Person=' in w[5] and not 'Poss=Yes' in w[5]:
				if token in ['ich', 'ihr', 'du', 'er', 'sie', 'es', 'wir', 'mich', 'mir', 'dich', 'dir', 'ihm', 'ihn',
				             'uns', 'ihnen']:
					count += 1
					matches.append(w[2].lower())
		elif lang == 'ru':
			if 'PRON' in w[3] and 'Person=' in w[5] and not 'Poss=Yes' in w[5]:
				if token in ['я', 'ты', 'вы', 'он', 'она', 'оно', 'мы', 'они', 'меня', 'тебя', 'его', 'её', 'ее',
				             'нас', 'вас', 'их', 'неё', 'нее', 'него', 'них', 'мне', 'тебе', 'ей', 'ему', 'нам', 'вам',
				             'им', 'ней', 'нему', 'ним', 'меня', 'тебя', 'него', 'мной', 'мною', 'тобой', 'тобою',
				             'Вами', 'им', 'ей', 'ею', 'нами', 'вами', 'ими', 'ним', 'нем', 'нём', 'ней', 'нею', 'ними']:
					count += 1
					matches.append(w[2].lower())
	return count, matches


def possdet(tree, lang):
	count = 0
	matches = []
	example = []
	for w in tree:
		lemma = w[2].lower()
		# own and eigen are not included as they do not compare to свой, it seems
		if lang == 'en':
			if lemma in ['my', 'your', 'his', 'her', 'its', 'our', 'their']:
				if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[5]:
					count += 1
					matches.append(w[2].lower())
		elif lang == 'de':
			if lemma in ['mein', 'dein', 'sein', 'ihr', 'Ihr|ihr', 'unser', 'eurer']:  # eurer does not occur
				if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[5]:
					count += 1
					matches.append(w[2].lower())
		elif lang == 'ru':
			if 'DET' in w[3] and lemma in ['мой', 'твой', 'ваш', 'его', 'ее', 'её', 'наш', 'их', 'ихний', 'свой']:
				count += 1
				matches.append(w[2].lower())
	return count, matches

# include only noun substituters, i.e. pronouns par excellence, of Indefinite, total and negative semantic subtypes
# recall va precision dilemma: how about including DET зачем-то? почему-то?
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
		elif lang == 'de':
			# 'PronType=Ind' in w[5]
			# This PronType is not annotated for the other two lang; German tags seem unreliable, they get man:378;
			# viel:145; einige:141; ander:120; solch:103; ein:95; mehr:69; manch:48; etwas:47; mehrere:35; was:23;
			# all:21; paar:12; welch:11; wenig:11; weniger:10; jemand:7; jeglich:3; irgendwelch:3; elektronenpaar:3;
			# diejenige:3; dergleichen:2; ebensowenig:2; irgendein:2; viele:2; derjenige:2; meist:2; soviel:2;
			if w[2] in ['etwas', 'irgendetwas', 'irgendwelch', 'irgendwas', 'jedermann', 'jedermanns', 'jemand',
			            'alles', 'niemand', 'nichts', 'irgendwo', 'manch'] and 'PronType=Ind' in w[5]:
				# einiges, nichts, vieles, manches, weniges, etwas (indefinite neuter pronouns)
				count += 1
				matches.append(w[2].lower())
		elif lang == 'ru':
			if w[2] in ['некто', 'нечто', 'нечего', 'никто', 'ничто', 'нигде', 'никуда', 'ниоткуда'] and w[3] == 'PRON':
				count += 1
				matches.append(w[2].lower())
			if re.search(r'-то|-нибудь|-либо', w[2], re.UNICODE) and 'какой' not in w[2]:
				'''
				какой-нибудь, любой и всякий учитываются в demdeterm
				'''
				# print(w[2])
				count += 1
				matches.append(w[2].lower())
			if re.match('кое', w[2], re.UNICODE) and 'какой' not in w[2]:
				# print(w[2])
				count += 1
				matches.append(w[2].lower())
			if w[2] in ['кто-кто', 'кого-кого', 'кому-кому', 'кем-кем', 'ком-ком', 'что-что',
			            'чего-чего', 'чему-чему', 'чем-чем', 'куда-куда', 'где-где']:
				count += 1
				matches.append(w[2].lower())
	return count, matches


## for now we use "all tokens receiving the tag (SCONJ, CCONJ) in the morphology annotation,
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
		if lang == 'de':
			if 'CCONJ' in w[3] and w[2] in ['und', 'oder', 'aber', 'sondern', 'sowie', 'als', 'wie', 'doch',
			                                'sowohl', 'denn', 'desto', 'noch', 'weder', 'entweder', 'bzw',
			                                'beziehungsweise', 'weshalb', 'und/oder', 'ob', 'woher', 'wenn',
			                                'jedoch', 'wofür', 'insbesondere', 'obwohl', 'um']:
				count += 1
				matches.append(w[2].lower())
		elif lang == 'ru':
			if 'CCONJ' in w[3] and w[2] in ['и', 'а', 'но', 'или', 'ни', 'да', 'причем', 'либо', 'зато', 'иначе',
			                                'только', 'ан', 'и/или', 'иль']:
				count += 1
				matches.append(w[2].lower())
	return count, matches


def sconj(tree, lang):
	# including that/dass/что as complimentizer (Bush announced that the United States would;
	# Und im Jahr 2003 kündigte Präsident Bush an, dass sich die Vereinigten Staaten dem Internationalen Thermonuklearen Experimentellen Reaktor anschließen würden
	count = 0
	matches = []
	for w in tree:
		if lang == 'en':
			if 'SCONJ' in w[3] and w[2] in ['that', 'if', 'as', 'of', 'while', 'because', 'by', 'for', 'to', 'than',
			                                'whether', 'in', 'about', 'before', 'after', 'on', 'with', 'from', 'like',
			                                'although', 'though', 'since', 'once', 'so', 'at', 'without', 'until',
			                                'into', 'despite', 'unless', 'whereas', 'over', 'upon', 'whilst', 'beyond',
			                                'towards', 'toward', 'but', 'except', 'cause', 'together']:
				count += 1
				matches.append(w[2].lower())
		
		elif lang == 'de':
			if 'SCONJ' in w[3] and w[2] in ['daß', 'wenn', 'dass', 'weil', 'da', 'ob', 'wie', 'als',
			                                'indem', 'während', 'obwohl', 'wobei', 'damit', 'bevor',
			                                'nachdem', 'sodass', 'denn', 'falls', 'bis', 'sobald',
			                                'solange', 'weshalb', 'ditzen', 'sofern', 'warum', 'obgleich',
			                                'zumal', 'sodaß', 'aber', 'wenngleich', 'wennen', 'wodurch',
			                                'wohingegen', 'ehe', 'worauf', 'seit', 'inwiefern', 'anstatt', 'der',
			                                'vordem', 'insofern', 'nahezu', 'wohl', 'manchmal', 'weilen', 'weiterhin',
			                                'doch', 'mit', 'gleichfalls']:
				count += 1
				matches.append(w[2].lower())
		
		elif lang == 'ru':
			if 'SCONJ' in w[3] and w[2] in ['что', 'как', 'если', 'чтобы', 'то', 'когда', 'чем', 'хотя', 'поскольку',
			                                'пока', 'тем', 'ведь', 'нежели', 'ибо', 'пусть', 'будто', 'словно', 'дабы',
			                                'раз', 'насколько', 'тот', 'коли', 'коль', 'хоть', 'разве', 'сколь',
			                                'ежели', 'покуда', 'постольку']:
				count += 1
				matches.append(w[2].lower())
	
	return count, matches


def whconj(tree, lang):
	# adverbial clause introduced by a pronominal ADV (not SCONJ)
	# why no что, кто? -- because they introduce relative clauses (see below)
	count = 0
	matches = []
	if tree[-1][2] != '?' and tree[-2][2] != '?':
		for w in tree:
			if lang == 'en':
				if w[2] in ['when', 'where', 'why']:  # why no who, what? --- they're not ADV
					count += 1
					matches.append(w[2].lower())
			if lang == 'de':
				if w[2] in ['wann', 'wo', 'warum']:
					count += 1
					matches.append(w[2].lower())
			if lang == 'ru':
				if w[2] in ['когда', 'где', 'куда', 'откуда', 'отчего', 'почему', 'зачем']:
					# 'какой' is a DET det
					count += 1
					matches.append(w[2].lower())
	return count, matches


# relativ, correl, pied
'''
returns:
-- all relative clauses, including correlative constructions and pied-piping construction
-- correlative dem PRON connected to subsequent CONJ (for EN they are only a subtype of relative causes,
   for DE/RU correlatives include clausal complements, introduced bu SCONJ);
   both types can be further complicated by piped preps ADP: о таком, о каком вы не слыхали
-- pied-piped: constractions with displaced (pied-piped) preposition
Examples of inconsistency in RU: Он настаивал на том, что (SCONJ) считал важным.
VS Его обвиняли в том, что (tagged SCONJ, but is PRON) он бросил школу.
'''


def relativ(tree, lang):
	tot_rel = 0
	matches = []  # this can be used to produce the freq dict of the relative PRON that introduce clauses
	pied = 0
	correlatives = 0
	'''
    CORRELATIVE clauses
    correlatives "указательное местоимение семантически и/или синтаксически связано
    с союзным словом в составе постпозитивного сентенциального актанта
    или относительного придаточного" (Пекелис, 2018)
    there are a lot of other types: Как только освоишься, так сразу напиши,
    Если понадоблюсь, то звони.
    I can't think of correlative constructions of non-relative type in English
    '''
	if tree[-1][2] != '?' and tree[-2][2] != '?':
		if lang == 'en':
			for w in tree:
				demonst_id = 0
				comma_id = 0
				if w[2] in ['which', 'that', 'whose', 'whom', 'what', 'who'] and w[3] == 'PRON':
					tot_rel += 1
					'''
                    25.3187 per 100 sents
                    this count excludes false positives when PRON is not introducing a clause
                    It 's as simple as that .
                    and
                    the use of the same PRON in non-affirmative sentences
                    '''
					matches.append(w[2].lower())
					correlat_id = w[0]
					for www in tree:
						# exclude total pronouns for now 'all','this', 'these','that',
						if www[3] in ['DET', 'PRON'] and www[2] in ['those', 'such']:
							demonst_id = www[0]
					if demonst_id != 0:
						if 0 < correlat_id - demonst_id <= 3:
							correlatives += 1
							'''
                            146, 0.7024 very rare phenomenon!
                            of those who voted for him
                            raising the living standards of those that are poor
                            only those who already have jobs
                            -- adding ,'that' extracts 10 more instances,
                            -- mostly errors (that is not what we are seeing)
                            '''
					kids_pos = get_kids_pos(w, tree)
					if kids_pos:
						'''
                        PIED-PIPING constructions: ex. patients for whom;
                        '''
						if 'ADP' in kids_pos:
							'''
                            unlike Russian code, tot_rel gets FPs
                            (i.e.phrasal occurences (can't use comma!)) to be excluded
                                ex. "English is an enabler of that"
                                But we will return to that later .
                                It 's as simple as that .
                            '''
							if w[6] < w[0]:
								'''
                                excluding cases when relative PRON has a dependent preposition and follows its head
                                '''
								tot_rel += -1
								# print()
								# print("NEGAT", ' '.join(w[1] for w in tree))
							else:
								pied += 1
								'''
                                197, 0.9477 per 100 sents, very rare!
                                speech in which he made this argument
                                the group against whom they had been fighting
                                technology for which Sony could take credit
                                '''
		#                                 print(tree[w[0]-2][1], w[1].upper(), tree[w[0]][1], file=sys.stderr)
		#                                 print("PIED", ' '.join(w[1] for w in tree), file=sys.stderr)
		
		elif lang == 'de':
			for w in tree:
				demonst_id = 0
				comma_id = 0
				'''
                entwickeln als ein fruchtbarer Streit darüber (PRON, conj),
                welche (PRON, det) Grundannahmen der Analyse...
                '''
				if (w[2] in ['der', 'welch', 'was', 'wer'] and w[3] == 'PRON') \
						or ('wo' in w[2] and 'PronType=Int,Rel' in w[5]):
					prev = get_prev(w, tree)
					correlat_id = w[0]
					for ww in tree:
						'''
                        allow up to 3 words between the comma and the Pron
                        '''
						if ww[1] == ',' and 0 < (correlat_id - ww[0]) <= 3:
							comma_id = ww[0]
							tot_rel += 1
							'''
                            23.6771 per 100 sents
                            general count for all types of relative clauses (non-existant for DE in UD)
                                zwei Regionen , in denen wir versucht haben
                                Oberflächenwasser , das sich in einem tiefen Loch ansammelt
                                viele andere Drogen , die abhängig machen
                                konnten wir sehr viel mehr Erfahrungswerte dahingehend sammeln,
                                welche Lösung für ..
                            '''
							#                             print(tree[w[0]-2][1], w[1].upper(), tree[w[0]][1], file=sys.stderr)
							#                             print(' '.join(w[1] for w in tree), file=sys.stderr)
							matches.append(w[2].lower())
							for www in tree:
								'''
                                leaving out jemand -> der; alles, einiges, nichts,
                                vieles, manches, weniges, etwas -> das; to be comparable to <those, wh. and such that>
                                and limit the distance btw comma and second part
                                '''
								if www[3] in ['DET', 'PRON'] and (0 < (comma_id - www[0]) <= 3) and 'PronType=Dem' in \
										www[5] and www[2] in ['darüber', 'der', 'das', 'dasselbe', 'jen',
										                      'solch', 'diejenigen', 'jener']:
									demonst_id = www[0]
									dem_tok = www[1]
					# PIED outside correlative constructions
					if demonst_id == 0 and comma_id != 0:
						if prev[3] == "ADP" or ('wo' in w[2] and 'PronType=Int,Rel' in w[5]):
							pied += 1
							'''
                            this count is not final
                            zu Spannungsabfall führt , wodurch sich die positiv geladenen .. neu anordnen
                            Der Prozeß , in dem aus den sich verteilenden
                            die Zelle , aus der ein Ei hervorgeht
                            '''
					#                             print(tree[w[0]-2][1], w[1].upper(), tree[w[0]][1])
					#                             print("PIED_prev_out", ' '.join(w[1] for w in tree))
					# CORRELATIVES
					if demonst_id != 0 and comma_id != 0:
						if demonst_id < comma_id < correlat_id:
							# print('CORR', ' '.join(w[1] for w in tree))  # this includes pied
							correlatives += 1  # including PIED
						# pied-piped constructions: скандал , в котором; трагедии , с которыми, в той конструкции , в какой она
						if (prev[3] == "ADP") or ('wo' in w[2] and 'PronType=Int,Rel' in w[
							5]):  # more pied, these are also inside correlative constructions
							'''
                            this gets just two instances
                            jenes Gebiet , in dem die Lösung des Rätsels liegen könnte
                            '''
							pied += 1
				#                             print("in_PIED_prev", ' '.join(w[1] for w in tree))
				
				# other correlative acl, что is not PRON : Er bestand darauf, was er für wichtig hielt.
				# for GE/RU this includes daruber, was; о том, что взять с собой (куда пойти учиться), non-existant in EN (?)
				elif (w[3] == 'SCONJ' and w[2] in ['dass', 'daß']) or w[2] == 'was':
					correlat_id = w[0]
					for ww in tree:
						if ww[1] == ',' and 0 < (correlat_id - ww[0]) < 2:
							comma_id = ww[0]
						for www in tree:
							'''
                            leaving out jemand -> der; alles, einiges, nichts, vieles, manches, weniges, etwas -> das;
                            '''
							if www[3] in ['DET', 'PRON'] and 0 < (comma_id - www[0]) <= 3 and 'PronType=Dem,Rel' in www[
								5]:
								'''
                                this grammatic value excludes 'dies'
                                and gets www[2] in ['darüber', 'der', 'das', 'dasselbe', 'jen', 'solch', 'diejenigen', 'jener']:
                                '''
								demonst_id = www[0]
					if demonst_id != 0 and comma_id != 0:
						if demonst_id < comma_id < correlat_id:
							#                             print(tree[w[0]-2][1], w[1].upper(), tree[w[0]][1])
							#                             print('type2_CORR', ' '.join(w[1] for w in tree))
							correlatives += 1  # alc including PIED Он настаивал на том, что считал важным.
							'''
                            2.3244
                            these are correlative clausal complements, NOT relative clauses!!
                                Dies weist darauf hin , daß durch
                                könnte darin bestehen , daß alle diese
                                Man geht in der Biologie davon aus , daß die Entwicklung
                                Eine Möglichkeit sehen Jaffe und seine Mitarbeiter darin , daß sich innerhalb
                            '''
							prev = get_prev(w, tree)  # better than using dependencies and siblings
							if prev[3] == "ADP":
								'''
                                final count for pied-piped constructions:
                                3.5237 per 100 sents
                                nothing gets here, but theoretically possible
                                '''
								pied += 1
		#                                 print("type2_PIED_prev", ' '.join(w[1] for w in tree))
		
		elif lang == 'ru':
			for w in tree:
				demonst_id = 0
				comma_id = 0
				# no PronType=Rel
				# кто and что occur in PRON obj (1) Те же , кого не лечили , были обречены (acl:relcl). and in (2) Все знали, кто этот человек (obl).
				# exclude DET as который_DET "у всех есть еще вот какая возможность"
				if (w[3] == 'PRON' and w[2] in ['который', 'что', 'кто']) or w[2] == 'какой':
					# CORRELATIVE тот/то/такой/весь/всё (N), (preposition) который/кто/что
					prev = get_prev(w, tree)
					correlat_id = w[0]
					for ww in tree:
						if ww[1] == ',' and 0 < (correlat_id - ww[0]) <= 3:
							comma_id = ww[0]
							tot_rel += 1  # placing after comma test excludes  мало кто, что угодно, который год, ни о каком разговоре
							matches.append(w[2].lower())
						# print("ALL", ' '.join(w[1] for w in tree))
						for www in tree:
							# do i need total pronouns 'весь', 'все', 'всё'?
							if www[3] in ['DET', 'PRON'] \
									and www[2] in ['тот', 'такой', 'то'] \
									and 0 < (comma_id - www[0]) <= 3:
								demonst_id = www[0]
					# PIED outside correlative constructions
					if demonst_id == 0 and comma_id != 0:
						prev = get_prev(w, tree)
						if prev[3] == "ADP":
							pied += 1
					# print("PIED_prev_out", ' '.join(w[1] for w in tree), file=sys.stderr)
					# CORRELATIVES
					if demonst_id != 0 and comma_id != 0:
						if demonst_id < comma_id < correlat_id:
							# print('CORR', ' '.join(w[1] for w in tree))  # this includes pied
							correlatives += 1  # including PIED
						# pied-piped constructions: скандал , в котором; трагедии , с которыми, в той конструкции , в какой она
						# also gets "изучить точки соприкосновения в том , что (PRON) касается борьбы с"
						if prev[3] == "ADP":  # more pied, these are also inside correlative constructions
							pied += 1
				# print("PIED_prev", ' '.join(w[1] for w in tree))
				
				# other correlative acl where что/dass is not PRON
				elif w[3] == 'SCONJ' and w[2] == 'что':
					correlat_id = w[0]
					for ww in tree:
						if ww[1] == ',' and 0 < (correlat_id - ww[0]) < 2:
							comma_id = ww[0]
						for www in tree:
							# leave out 'весь', 'все', 'всё',
							if www[3] in ['DET', 'PRON'] \
									and www[2] in ['тот', 'такой', 'то'] \
									and 0 < (comma_id - www[0]) <= 4:
								demonst_id = www[0]
					if demonst_id != 0 and comma_id != 0:
						if demonst_id < comma_id < correlat_id:
							# print('type2_CORR', ' '.join(w[1] for w in tree))
							correlatives += 1  # alc including PIED Он настаивал на том, что считал важным.
							prev = get_prev(w, tree)
							if prev[3] == "ADP":
								pied += 1
		# print("type2_PIED_prev", ' '.join(w[1] for w in tree))
	
	return tot_rel, matches, pied, correlatives


def word_length(tree):
	words = 0
	letters = 0
	for el in tree:
		if not el[1] in ['.', ',', '!', '?', ':', ';', '"', '-', '—', '(', ')']:
			words += 1
			for let in el[1]:
				letters += 1
		else:
			continue
	if words == 0:
		print(tree)
	av_wordlength = letters / words
	# print(letters, words, file=sys.stderr)
	return av_wordlength


def copulas(tree):
	copCount = 0
	for i, w in enumerate(tree):
		if w[7] == "cop" and w[2] in ['be', 'sein', 'быть', 'это']:
			for prev_w in [tree[i - 1], tree[i - 2], tree[i - 3]]:
				if prev_w[2] == 'there':
					copCount += -1
					continue
			copCount += 1
	
	return copCount


# this feature is fishy, because at text processing stage I delete all sents that are less than 3 words long and many of them are interrogative
def interrog(tree, lang):
	count = 0
	matches = []
	if lang in ['en', 'de', 'ru']:
		last = tree[-1]
		lastbut = tree[-2]
		if tree[-1][2] == '?' or tree[-2][2] == '?':
			count += 1
			matches.append(last[2])
			matches.append(lastbut[2])
	
	return count, matches


def nn(tree, lang):
	count = 0
	matches = []
	if lang in ['en', 'de', 'ru']:
		for w in tree:
			lemma = w[2].lower()
			if 'NOUN' in w[3]:
				count += 1
				matches.append(lemma)
	# if lang == 'de':
	# 	print(w[3])
	
	return count, matches


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


## graph-based feature

## this function tests connectedness of the graph; it is called from skeakdiff below and helps to skip malformed sentences
## additionally it produces the sentence graph (NB!! it slows down the process, hence if graphs are needed call a separate function)
def test_sanity(tree):
	ARPACKOptions.maxiter = 3000
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
	
	# print(bad_trees)
	
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


### this is a slower function which allows to print graphs for sentences (to be used in overall_freqs)
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


# exclude PUNCT, SYM, X from tokens and adjust counts of content items for Russian by excluding modal verbs and adverbs
def lex_ty_to(tree, lang):
	lex_types = []
	lex_tokens = []
	for w in tree:
		if w[3] in ['PUNCT', 'SYM', 'X']:
			continue
		if lang == 'en' or lang == 'de':
			if 'ADJ' in w[3] or 'ADV' in w[3] or 'VERB' in w[3] or 'NOUN' in w[3]:
				lex_type = w[2] + '_' + w[3]
				lex_types.append(lex_type)
				lex_token = w[1] + '_' + w[3]
				lex_tokens.append(lex_token)
		if lang == 'ru':
			# for RU we exclude 2 modal verbs and 3 modal adverbs to offset the EN/GE modal verbs
			# that are tagged as AUX (i.e. not content words, but functionals)
			if w[2] == 'мочь':
				# print(' '.join(w[1] for w in tree))
				continue
			if w[2] == 'следовать':  # haveto, should
				kids_pos = get_kids_pos(w, tree)
				kids_gr = get_kids_feats(w, tree)
				if 'VERB' in kids_pos and 'VerbForm=Inf' in kids_gr:
					# print(' '.join(w[1] for w in tree), file=sys.stderr)
					continue
			# 3 modal adverbs
			if w[2] == 'можно' or w[2] == 'нельзя' or w[2] == 'надо':
				continue
			if 'ADJ' in w[3] or 'ADV' in w[3] or 'VERB' in w[3] or 'NOUN' in w[3]:
				lex_type = w[2] + '_' + w[3]
				lex_types.append(lex_type)
				lex_token = w[1] + '_' + w[3]
				lex_tokens.append(lex_token)
	
	lex_types = len(set(lex_types))
	lex_tokens = len(lex_tokens)
	
	return lex_types, lex_tokens


def modpred(tree, lang, mpred_dic):
	counter_can = 0
	counter_haveto = 0
	counter_adj = 0
	counter_adv = 0
	mpred = 0
	matches = []
	if lang == 'en':
		mpred_lst = mpred_dic[lang]
		for w in tree:
			if w[4] == 'MD' and w[2] != 'will' and w[2] != 'shall':
				mpred += 1
				matches.append(w[2])
			if w[2] in mpred_lst:
				kids_pos = get_kids_pos(w, tree)
				if 'AUX' in kids_pos:
					counter_adj += 1
					matches.append(w[2])
			if w[2] == 'have' and w[3] != 'AUX':
				own_id = w[0]
				inf_kid_id = choose_kid_by_posfeat(w, tree, 'VERB', 'VerbForm=Inf')
				'''
				limit the distance between have and inf to avoid getting complex sentences with have and modal predicate
				I do have friends at home who can not survive without a nap .
				'''
				if inf_kid_id != None and abs(own_id - inf_kid_id) < 4:
					causative1 = choose_kid_by_posrel(w, tree, 'NOUN', 'obj')
					if causative1 != None and causative1 < inf_kid_id:
						'''
						this set of rules gets:
						you have time to practise more
						have a colleague throw another ball onto the table
						'''
						continue
					else:
						counter_haveto += 1
						matches.append(w[2])
		# print(' '.join(w[1] for w in tree),file=sys.stderr)
		mpred = mpred + counter_haveto + counter_adj
	
	if lang == 'de':
		mpred_lst = mpred_dic[lang]
		for w in tree:
			if 'VM' in w[4] and w[2] in ['dürfen', 'können', 'mögen', 'müssen', 'sollen', 'wollen']:
				mpred += 1
				matches.append(w[2])
			# print(w[2])
			# print(' '.join(w[1] for w in tree))
			# this gets cases that are misspelt and mislemmatized
			elif 'VM' in w[4] and not w[2] in ['dürfen', 'können', 'mögen', 'müssen', 'sollen', 'wollen']:
				mpred += 1
				matches.append(w[2])
			# print(w[2])
			# print("MISSLEMMA", ' '.join(w[1] for w in tree))
			# gets misstagged modals
			elif not 'VM' in w[4] and w[2] in ['dürfen', 'können', 'mögen', 'müssen', 'sollen', 'wollen']:
				# print(w[2], w[3])
				# print("MISTAGGED", ' '.join(w[1] for w in tree))
				mpred += 1
				matches.append(w[2])
			
			### haben, sein + zu + infinitive
			### Es ist noch viel zu tun. see http://canoo.net/services/OnlineGrammar/Wort/Verb/VollHilfModal/haben-sein.html
			###  Ihr habt euch nicht hier aufzuhalten. Gets a lot of errors and is very infrequent
			# if w[2] == 'haben' and w[3] != 'AUX':
			# 	kids_feats = get_kids_feats(w, tree)
			# 	kids_pos = get_kids_pos(w, tree)
			# 	if 'VerbForm=Inf' in kids_feats and not 'NOUN' in kids_pos:
			# 		print(' '.join(w[1] for w in tree), file=sys.stderr)
			# 		mpred += 1
			
			elif w[2] in mpred_lst:
				'''
				Es ist jedoch offensichtlich, dass weiterhin Druck ausgeübt wird.
				Es ist unmöglich abzulehnen.
				Selbstreparatur ist immerhin möglich .
				Natürlich ist es notwendig , Europas Wettbewerbsfähigkeit zu verbessern
				Es ist klar , dass
				Auch auf dem Arbeitsmarkt sind zusätzliche Impulse notwendig .
				In dem Gespräch mit den Menschen wurde uns klar , daß
				'''
				kids_lem = get_kids_lem(w, tree)
				if 'sein' in kids_lem or 'werden' in kids_lem:
					counter_adj += 1
		### After a lot of hesitation I am excluding sein/be + inf from the modal verb realm treating Inf as a predicative in these cases
		# it is also not excluded from Inf counts on the assumption that it is not forced use of Inf
		# elif w[2] == 'sein' and w[3] == 'AUX':
		# 	'''
		# 	It looks like a blend of modal and passive
		# 	Es ist noch viel zu tun.
		# 	Einige Erfolge sind ansatzweise bereits zu verzeichnen .
		# 	Dieses Experiment ist zwar ein wichtiger Meilenstein der Stammzellforschung, zeigt aber in seinen Details, wie viele Probleme noch zu lösen sind.
		# 	Auf den Fotos ist zu sehen, daß unterschiedlich starke Schmelzprozesse die Oberflächen umgestalteten .
		# 	'''
		# 	head = get_headwd(w, tree)
		# 	if head and 'VerbForm=Inf' in head[5]:
		# 		# print(' '.join(w[1] for w in tree), file=sys.stderr)
		# 		sein_mpred += 1
		'''
		other modal verbs -- verstehen, pflegen, drohen, pflegen scheinen -- are not included;
		they are omitted from analysis for other langs, too
		
		DONE TODO что-то я не нашла в немецком модальных структур типа:
			RU: в Америке должны наконец задаться вопросом, как ни банально это звучит ,...желая мира, нужно готовиться к войне .
				Бороться с националистическим подтекстом, доказывать вину этих уродов - жизненно необходимо
				DE: ... sind von entscheidender Bedeutung.
			EN: But with all that said, I'm not sure Putin is panicking.
					Trotzdem bin ich mir nicht sicher, ob Putin in Panik gerät.
				It is obvious, however, that pressure continues to be applied
					Es ist jedoch offensichtlich, dass weiterhin Druck ausgeübt wird
				And shame is likely what Trump supporters will feel if he wins .
				He is likely to decline. -- Er wird wahrscheinlich ablehnen.
		Если такие есть нужен список соответствующих прилагательных (см. EN-RU списки)
		'''
		mpred = mpred + counter_adj  # + sein_mpred
	if lang == 'ru':
		mpred_lst = mpred_dic[lang]
		for w in tree:
			# 2 verbs
			if w[2] == 'мочь':
				counter_can += 1
				matches.append(w[2])
			if w[2] == 'следовать':
				kids_pos = get_kids_pos(w, tree)
				kids_gr = get_kids_feats(w, tree)
				if 'VERB' in kids_pos and 'VerbForm=Inf' in kids_gr:
					counter_haveto += 1
					matches.append(w[2])
			# 3 modal adverbs
			if w[2] == 'можно' or w[2] == 'нельзя' or w[2] == 'надо':
				counter_adv += 1
				matches.append(w[2])
			# 11 listed short ADJ
			if w[2] in mpred_lst and 'Variant=Short' in w[5]:
				counter_adj += 1
				matches.append(w[2])
		# print("modalADJ", ' '.join(w[1] for w in tree))
		mpred = counter_can + counter_haveto + counter_adj + counter_adv
	
	return mpred, matches


def advquantif(tree, lang, madv_dic):
	mod_quantif = 0
	matches = []
	for w in tree:
		if lang == 'en':
			madv_lst = madv_dic[lang]
			if w[2] in madv_lst and w[3] == 'ADV':
				mod_quantif += 1
				matches.append(w[2])
		if lang == 'de':
			madv_lst = madv_dic[lang]
			if w[2] in madv_lst and w[3] == 'ADV':
				head = get_headwd(w, tree)
				if head:
					if head[3] == 'NOUN':
						continue
					else:
						mod_quantif += 1
						matches.append(w[2])
		if lang == 'ru':
			madv_lst = madv_dic[lang]
			non_ADVquantif = ['еле', 'очень', 'вшестеро', 'невыразимо', 'излишне',
			                  'еле-еле', 'чуть-чуть', 'едва-едва',
			                  'только', 'капельку', 'чуточку', 'едва']
			if w[2] in madv_lst and w[3] == 'ADV':
				mod_quantif += 1
				matches.append(w[2])
			if w[1] in non_ADVquantif:  # based on token, not lemma
				mod_quantif += 1
				matches.append(w[2])
	
	return mod_quantif, matches


def finites(tree, lang):
	fins = 0
	matches = []
	inf = 0
	if lang == 'en' or lang == 'ru':
		for w in tree:
			if 'VerbForm=Fin' in w[5]:
				fins += 1
	if lang == 'de':
		for w in tree:
			kids_lem = get_kids_lem(w, tree)
			mv = ['können', 'müssen', 'sollen', 'wollen', 'mögen', 'dürfen', 'konnen', 'mußen']
			mv0 = set(mv)
			kids_lem0 = set(kids_lem)
			# excludes Grundsätzlich sind alle Arbeitnehmer in der Arbeitslosenversicherung versichert (Fin) (around 140 cases)
			if 'VerbForm=Fin' in w[5]:
				# in cases below the verb can't be in finite form but is tagged as such: daß das Nervenwachstum schon durch kleinste elektrische Felder mit einer Stärke von nur wenigen Millivolt pro Millimeter beeinflußt (Fin) wird .
				if ('sein' and 'werden') in kids_lem:
					continue  # the actual VerbForm here is either Inf or Part
				elif len(mv0.intersection(kids_lem0)) != 0:
					continue  # infs += 1
				# the rule testing whether the correct Inf VerbForm occurs in  accusativus cum infinitivo returns two TP and three FP and is complicated
				else:
					fins += 1
					matches.append(w[2])
			if 'VerbForm=Part' in w[5]:
				kids_pos = get_kids_pos(w, tree)
				kids_lem = get_kids_lem(w, tree)
				kids_rel = get_kids_rel(w, tree)
				# exclude analytical forms, double-check for unrecognized auxilaries (63 errors like that)
				if not "AUX" in kids_pos and not 'sein' in kids_lem \
						and not 'werden' in kids_lem and not 'haben' in kids_lem:
					# also exclude errors where in the absence of an aux, there is a recognized nsubj dependency, which makes it a finite form rather than a Part
					# Mit dieser Frage verlassen (Part!?) wir (nsubj) das Gebiet zellinterner Veränderungen und wenden uns dem extrazellulaeren Bereich zu .
					# gets 297 cases like: Auf der Haut von Fröschen treten Elektropotentiale sogar schon vor der Entwicklung von Nerven auf , wobei das Innere des Embryo eine positive Ladung aufweist .
					if "nsubj" in kids_rel:
						fins += 1
						matches.append(w[2])
	
	return fins, matches


## 4-6 July: infinitives, participle, nominatives, passives done for three languages
def infinitives(tree, lang, mpred_dic):
	infs = 0
	'''
	General approach for EN/DE:
	get all Inf with particle, excluding after modal phrases + cases of true bare inf (but not analytical forms or modal verbs)
	'''
	if lang == 'en':
		'''
		get all to-inf excluding after have-to, going-to and modal phrases
		+ true bare inf (inc. causatives and perception)
		returns : 7098 cases

		an alternative approach :
		exclude cases
			after MD(modal aux, inc. will/shall)
			after dependent of modal phrases,
			inc. have to and going to, to be likely to V-Inf
		returns 11943 cases
		'''
		for w in tree:
			if 'VerbForm=Inf' in w[5]:
				own_id = w[0]
				head = get_headwd(w, tree)
				if choose_kid_by_lempos(w, tree, 'to', 'PART') != None:  # cases of to-inf
					if head:
						if head[2] == 'have':
							objective1 = choose_kid_by_posrel(head, tree, 'NOUN', 'obj')
							#                         print(' '.join(w[1] for w in tree))
							if objective1 != None and objective1 < own_id:
								'''
								while excluding true have to V-Inf, retain causative constructions: to have the euro replace the dollar
								and Complex Objects:
									you have so many places to eat
									have a duty to protect the weakest in society
								34 cases
								'''
								infs += 1
							else:
								'''
								People who have to fake emotions at work , burn out faster
								'''
								continue
						elif head[2] == 'go' or head[2] in mpred_dic[lang]:
							'''
							ex. we 're likely to see more exciting findings
							It 's dark in the cabin ; someone is going to step on your face !
							They 're going to have to .
							Bush is not going to withdraw the troops .
							425 cases excluded
							'''
							continue
						else:
							'''
							classic to-inf with head:
							I think if we decided to make French the company 's global language we would have had a revolt on our hands .
							They do n't want to lose face by using the wrong word
							6467 cases
							'''
							#                                 print(w[1].upper())
							#                                 print(' '.join(w[1] for w in tree))
							infs += 1
					else:
						'''
						classic to-inf without head:
						When in doubt , it 's best to ask .
						Or , to put it another way , maybe it 's selective laziness .
						How to explain this cleanliness and punctuality ?
						This is not to write the whole film off .
						How are we to trust them ?
						69 cases
						'''
						infs += 1
				#                     BARE Inf cases:
				elif w[3] != 'AUX' and head and head[2] in ['help', 'make', 'bid', 'let', 'see', 'hear',
				                                            'watch', 'dare', 'feel', 'have']:
					'''
					the first bit excludes cases like:
					we would have (Inf, dependent on have) had a revolt on our hands .

					gets 528 good cases like:
					you could see their culture come (Inf) to life .
					to do anything to have Russia pay (Inf) a price for its aggressive behavior
					Let 's start across the Atlantic
					'''
					infs += 1
	
	if lang == 'de':
		'''
		German rules:
		all zu-Inf, excluding after modal phrases: Es ist notwendig zu sagen; this supposedly gets all true Infs
		all bare inf dependent on
			hören, sehen, spüren
			lassen, gehen, bleiben, helfen, lehren
		The alternative approach of filtering analytical forms and modals
		and ALL Fins mistagged as Inf and misparsed sentences returns:
		1164 cases of Infs
		'''
		for w in tree:
			if 'VerbForm=Inf' in w[5]:
				head = get_headwd(w, tree)
				'''
				we have to exclude true negatives, mostly finite forms.
				This does not help with false positives
				'''
				if choose_kid_by_lempos(w, tree, 'zu', 'PART') != None:  # cases of zu-inf
					if head and head[2] in mpred_dic[lang]:
						'''
						exclude a few unwanted zu-Inf after modal predicates
							ex.: Alle Bordinstrumente werden notwendig sein, um die Antworten auf diese Fragen zu finden .
							ex. In jedem Fall ist es notwendig , vor der Nachrichtenübermittlung einen Schlüssel zu vereinbaren .
						'''
						continue
					else:
						'''
						1160 cases
						Diese Energie ist erforderlich , um die Molekülpaare wieder zu trennen .
						Derzeit ist man dabei , die verschiedenen Varianten genauer zu untersuchen , und bisher scheinen sie größtenteils die gleichen Funktionen zu haben .
						'''
						infs += 1
				
				elif head and head[2] in ['hören', 'sehen', 'spüren', 'lassen', 'gehen', 'bleiben', 'helfen', 'lehren']:
					'''
					+158 cases like
					Es bleibt zu erforschen , wie nützlich oder wie schädlich es sich auf die lebende Zelle auswirken kann .
					'''
					infs += 1
	
	if lang == 'ru':
		modal_preds = modpred(tree, lang, mpred_dic)[0]
		for w in tree:
			# exclude: "пока Россия будет проводить агрессивную политику", "отношения будут ухудшаться"
			if 'VerbForm=Inf' in w[5]:
				if has_auxkid_by_lem(w, tree, 'быть') != True:
					#                     print(' '.join(w[1] for w in tree))
					infs += 1
		infs = infs - modal_preds
	
	return infs


def participles(tree, lang):
	partconv = 0
	#     error_counter = 0
	for w in tree:
		if lang == 'en':
			'''
			Participial constructions (both VBN and VBG) VerbForm=Part + acl gets all participial clauses! exclude AUX in dependents to get WHIZ deletion relatives
			it appears that VBG is tagged as Part only as part of analytical predicate:
			all traditional Participial constructions are tagged Ger as well as cases like "the ongoing troubles":
			EXAMPLES:
			incidents such as a husband strangling his wife,
			I 'm not going to talk to you about intelligence (to exclude that!)
			adding that many are saying they see planning happening for a terrorist attack. (2 Part, 3 Ger)
			'''
			if 'VerbForm=Part' in w[5] and not "amod" in w[7]:
				'''
				this gets only VBN, ing-Part is exluded by no-AUX rule below
				the last bit excludes: it's not a done thing
				'''
				head = get_headwd(w, tree)
				if head:
					head_kids_pos = get_kids_pos(head, tree)
					num_aux = head_kids_pos.count('AUX')
					if num_aux >= 2 and w[1] == 'been':
						'''
						exclude 'been' (VerbForm=Part) in has been adopted/has been willing/has been instrumental
						(709 cases)
						'''
						#                             error_counter += 1
						#                             print(w[1].upper(), head[1])
						#                             print(' '.join(w[1] for w in tree))
						continue
				kids_pos = get_kids_pos(w, tree)
				if not "AUX" in kids_pos:
					'''
					this excludes all analytical forms to get
					fabrications created (Part) by those who would see Britain isolated (Part) from
					'''
					partconv += 1
			# Gerunds and constructions VerbForm=Ger in all functions, but amod, to exclude near adjectival uses like growing debt, following principle
			if 'VerbForm=Ger' in w[5] and not "amod" in w[7]:
				'''
				After years of translating emails , webinars and other materials
				'''
				head = get_headwd(w, tree)
				if head:
					head_kids_pos = get_kids_pos(head, tree)
					num_aux = head_kids_pos.count('AUX')
					if num_aux >= 2 and w[1] == 'being':
						'''
						exclude 'being' (VerbForm=Ger) in is being adopted (107 cases)
						'''
						continue
				else:
					partconv += 1
		
		if lang == 'de':
			
			'''
			fixing tagging errors with Part tagged as (1) Inf and as (2) Fin
			in the presence of dependent sein or haben as head respectively

			'''
			if 'VerbForm=Part' in w[5] and not "amod" in w[7]:
				'''
				AUSGEHANDELTEN
				Aber eine Kontroverse über die Umsetzung der (determiner in Gen Case?) in dem vergangenen (ADJ) Jahr
				ausgehandelten entscheidenden (ADJ) marktgestützten (ADJ) Bestimmungen droht ,
				die kontinuierlichen Fortschritte zu stören ,
				die jedermanns Ziel sein sollten .
				'''
				kids_pos = get_kids_pos(w, tree)
				kids_lem = get_kids_lem(w, tree)
				kids_rel = get_kids_rel(w, tree)
				head = get_headwd(w, tree)
				if "AUX" in kids_pos or 'sein' in kids_lem:  # this excludes all analytical forms
					'''
					2403 cases of 3040
					sein + Part with sein mistagged as VERB rather than AUX
					Doch war (VERB, 13, cop) die Überprüfung dieser Annahme zunächst
					an den Grenzen der Technik gescheitert .
					'''
					continue
				elif 'nsubj' in kids_rel or 'nsubj:pass' in kids_rel:
					'''
					309 cases
					mistagged Part, when must be Fin because have an nsubj dependent
						ex. Mit dieser Frage verlassen (Part?) wir das Gebiet
						zellinterner Veränderungen und wenden uns dem extrazellulaeren Bereich zu .
						ex. Aendert man die Richtung des Lichts, so ändert sich auch die Stelle
						des Stromzuflusses , wobei die Richtung des Stroms immer die Richtung
						des Wachstums bestimmt (Part?) .
					'''
					
					continue
				elif head:
					head_kids_lem = get_kids_lem(head, tree)
					if 'haben' in head_kids_lem or 'werden' in head_kids_lem:
						'''
						unrecognized Present Perfect Tense:
							Daher haben (aux, 4) auch diejenigen, die - oft von anderen - als "Afrikaner" etikettiert werden,
							sich von diesem Afrika-Bild gelöst (Part, 4) .
							Einst war (aux,4) sie erfunden worden (aux,4) für den Nationalstaat.
						captures coordinated Part:
							gezeigt in Wir haben unseren Willen und unsere Unterstützung für den Euro-Beitritt bekräftigt und gezeigt , dass
						'''
						continue
				
				else:
					partconv += 1
			
			if w[4] == 'ADJD' and w[1].endswith('d') and (w[7] == 'advmod' or w[7] == 'acl'):
				head_conv = get_headwd(w, tree)
				if head_conv and head_conv[3] == 'VERB':
					'''
					big problem: Present Participles are not recognized they are all ADJ in advmod
						ex. Der Hund stand bellend (ADJ, 3, advmod) am Fenster.
					solution: count in ADJ in advmod function, ending in -d dependent on a VERB (82 cases), otherwise I get ähnlich and genau
					New solution: use the native tag ADJD to get participial construction too (93 cases)

					Anscheinend steckt darin das Arsen .
					Viele finden es extrem spannend, ein virtuelles Unternehmen zu gründen und möchten es nachahmen.
					There are no Im Garten spielend, sang das Kind. examples found!

					German uses present participles primarily as adjectives and adverbs, not as verbs.
					English present tense, “he is running,” etc., is expressed in German with the present tense:
					er läuft, sie schwimmt.
					'''
					partconv += 1
		if lang == 'ru':
			if 'VerbForm=Part' in w[5] and 'Variant=Short' not in w[5] and "amod" not in w[7]:
				kids_pos = get_kids_pos(w, tree)
				if "AUX" not in kids_pos:
					partconv += 1
			if 'VerbForm=Conv' in w[5]:
				kids_pos = get_kids_pos(w, tree)
				if "AUX" not in kids_pos:
					partconv += 1
	#                     print(w[1].upper())
	#                     print(' '.join(w[1] for w in tree))
	return partconv


## this is the enhanced function that expects a stoplist and a list of approved V>N converts
def nominals(trees, lang, stop_dict, deverbs):
	res = 0
	stoplst = stop_dict[lang]
	converted = deverbs[lang]
	for tree in trees:
		for w in tree:
			if lang == 'en':
				if w[2] not in stoplst:
					if 'NOUN' in w[3] and (w[2].endswith('ment') or w[2].endswith('tion')):
						res += 1
				
				if 'NOUN' in w[3] and w[2] in converted:  # this is a filtered golist from our data
					if w[1].endswith('ing'):
						continue
					kids_pos = get_kids_pos(w, tree)
					if 'DET' not in kids_pos and 'ADJ' not in kids_pos and 'ADJ' not in kids_pos:
						if 'Number=Sing' in w[5]:
							continue
						else:
							res += 1
					else:
						res += 1
			
			if lang == 'de':
				if w[2] not in stoplst:
					if 'NOUN' in w[3] and (w[2].endswith('ung') or w[2].endswith('tion')):
						res += 1
				
				if 'NOUN' in w[3] and w[2] in converted:  # this is a filtered golist from our data
					res += 1
			
			if lang == 'ru':
				if w[2] not in stoplst and 'NOUN' in w[3] \
						and (w[2].endswith('тие') or w[2].endswith('ение')
						     or w[2].endswith('ание') or w[2].endswith('ство')
						     or w[2].endswith('ция') or w[2].endswith('ота')) \
						and 'Number=Plur' not in w[5]:
					res += 1
	return res


def passives(tree, lang):
	'''
	for all languages we are counting only passive predicated and
	don't account for adjectival/relative clauses with VBN!
	сигналы, испускаемые; исследований, проведенных

	long passives account only for Animate agents (we count in only by/von phrases and Case=Ins+Animacy=Anim)
	'''
	allpass = 0
	by_pass = 0
	counter_sem = 0
	if lang == 'en':
		for w in tree:
			'''
			basic rule
				aux:pass dependent on w with Voice=Pass in rel
			'''
			if w[7] == 'aux:pass':
				head = get_headwd(w, tree)
				if head and 'Voice=Pass' in head[5]:
					'''
					3196, 15.374994
					'''
					allpass += 1
					obl_kids_ids = []
					for i in ['NOUN', 'PRON', 'PROPN']:
						'''
						351 bypassives against 245 for only NOUNs
						'''
						obl_kid_id = choose_kid_by_posrel(head, tree, i, 'obl')
						if obl_kid_id:
							obl_kids_ids.append(obl_kid_id)
					
					if obl_kids_ids:
						for i in obl_kids_ids:
							by_grandchild = choose_kid_by_lempos(tree[i], tree, 'by', 'ADP')
							if by_grandchild:  ### Animacy is not a gr feature in English
								by_pass += 1
							#                           print(w[1].upper(), head[1].upper(), tree[by_grandchild][1], tree[obl_kid_id][1])
							#                             print(' '.join(w[1] for w in tree))
							else:
								'''
								i cannot count agentless here because it is possible to count one passive twice
								'''
								continue
					else:
						continue
		
		agentless = allpass - by_pass
	
	if lang == 'de':
		'''
		(1)`Vorgangspassiv’:
		Voice=Pass (прикреплен на вспомогательном глаголе aux:pass) к нему есть глагол VerbForm=Part
			NB! Voice=Pass is not among tagged features in German!!
		(2) `Zustandspassiv’:
		AUX (с drel aux:pass)+VerbForm=Part
		NB! there is no formal difference between the two if we rely on the tags

		50a Ein kleinerer , doch ebenso beständiger Anteil wird von Kalziumionen getragen .
		Doch konzentrierte sich die Arbeit in solchen Labors auf Zellen , die wie die Nervenzelle für die Übertragung elektrischer Signale bereits voll ausgebildet sind .
		'''
		for w in tree:
			if w[7] == 'aux:pass':
				'''
				the approach consistent with English returns:
				1401, 17.321958
				while if we do

				if 'VerbForm=Part' in w[5]:
					passaux_kid_id = choose_kid_by_posrel(w,tree,'AUX','aux:pass')
					if passaux_kid_id:
				we get 1364, 16.864491
				'''
				head = get_headwd(w, tree)
				if head and 'VerbForm=Part' in head[5]:
					allpass += 1
					obl_kids_ids = []
					for i in ['NOUN', 'PRON', 'PROPN']:
						'''
						120 bypassives
						Diese Auffassung wird neuerdings von Paul Giguere von der Laval University in Quebec bestritten .
						Sie wurden von den Genen abgelöst , die sämtliche Grundmerkmale annahmen , die zuvor die Lebewesen kennzeichneten .
						'''
						obl_kid_id = choose_kid_by_posrel(head, tree, i, 'obl')
						if obl_kid_id:
							obl_kids_ids.append(obl_kid_id)
					
					if obl_kids_ids:
						for i in obl_kids_ids:
							by_grandchild = choose_kid_by_lempos(tree[i], tree, 'von', 'ADP')
							if by_grandchild:  ### Animacy is not a gr feature in German
								by_pass += 1
							
							else:
								continue
					else:
						continue
			if w[2] == 'lassen' and w[3] == 'VERB':
				
				'''
				(3) ‘passive-like’: lassen sich + VerbForm=Inf
				53 cases
				Die Qualität einer bestimmten Option läßt sich nicht isoliert von ihren Alternativen bewerten .
				Vor allem haben die Briten jedoch eine Vision , die sich realisieren lässt .
				'''
				sich_kid_id = choose_kid_by_posfeat(w, tree, 'PRON', 'PronType=Prs|Reflex=Yes')
				inf_kid_id = choose_kid_by_posfeat(w, tree, 'VERB', 'VerbForm=Inf')
				if sich_kid_id and inf_kid_id:
					counter_sem += 1
		agentless = allpass - by_pass + counter_sem
	
	if lang == 'ru':
		for w in tree:
			if 'Variant=Short|VerbForm=Part|Voice=Pass' in w[5] or 'VerbForm=Fin|Voice=Pass' in w[5]:
				'''
				21556, 15.575820
				политика была направлена
				сливки будут сняты
				война велась
				риторика накаляется
				средства выделяются
				'''
				allpass += 1
				obl_kids_ids = []
				for i in ['NOUN', 'PRON', 'PROPN']:
					obl_kid_id = choose_kid_by_posrel(w, tree, i, 'obl')
					if obl_kid_id:
						obl_kids_ids.append(obl_kid_id)
				
				if obl_kids_ids:
					for i in obl_kids_ids:
						try:
							if 'Case=Ins' in tree[i][5] and tree[i][2] not in ['образ', 'лето', 'осень', 'зима',
							                                                   'весна', 'утро', 'вечер', 'ночь']:
								kids_pos = get_kids_pos(tree[i], tree)
								if kids_pos and not 'ADP' in kids_pos:
									'''
									filter out prepositional Ins
										предприняты c целью obl
										связано c лоббированием
									but
									this still gets a lot of noise (1734 cases):
										нажито нечестным путем (Case=Ins)
										признавалось орудием (Case=Ins) преступления
									'''
									if tree[i][5].split('|')[0] == 'Animacy=Anim':
										'''
										down to 283 cases, some are still ambiguous and it excludes collective nouns (ex. населением):
											избран первым президентом Всемирного конгресса
										была завербована Астемировым и Горчхановым
										места на нем могут быть заняты другими конкурентами
										охраняется сотрудниками правоохранительных органов
										'''
										by_pass += 1
								else:
									continue
						except IndexError:
							continue
				
				else:
					continue
			
			#             elif 'VerbForm=Part|Voice=Pass' in w[5] and 'amod' not in w[7]:
			#                 '''
			#                 this gets passive participial constractions, inc. pre-posed, which we decided to forgo
			#                 Но нелегально произведенная смесь , купленная в местном магазинчике ,
			#                 Он напомнил о недавно перечисленных Палестине $ 10 млн .
			#                 If we count them in, our passives freq sour to 31624, 22.850702
			#                 '''
			#                 allpass += 1
			#                 obl_kids_ids = []
			# #                 print(w[1].upper())
			# #                 print(' '.join(w[1] for w in tree))
			#                 for i in ['NOUN','PRON','PROPN']:
			#                     '''
			
			#                     '''
			#                     obl_kid_id = choose_kid_by_posrel(w, tree, i, 'obl')
			#                     if obl_kid_id:
			#                         obl_kids_ids.append(obl_kid_id)
			
			#                 if obl_kids_ids:
			#                     for i in obl_kids_ids:
			#                         if 'Case=Ins' in tree[i][5] and tree[i][2] not in ['образ', 'лето', 'осень', 'зима', 'весна', 'утро', 'вечер', 'ночь']:
			#                             kids_pos = get_kids_pos(tree[i],tree)
			#                             if kids_pos and not 'ADP' in kids_pos:
			#                                 '''
			#                                 filter out prepositional Ins
			#                                 предприняты c целью obl
			#                                 связано c лоббированием
			#                                 '''
			#                                 if tree[i][5].split('|')[0] == 'Animacy=Anim':
			#                                     '''
			#                                     634-283 = 351
			#                                     '''
			#                                     by_pass += 1
			# #                                     print(w[1].upper(), tree[i][1].upper(), tree[i][5].split('|')[0])
			# #                                     print(' '.join(w[1] for w in tree))
			#                         else:
			#                             continue
			#                 else:
			#                     continue
			
			if 'VERB' in w[3] and 'Number=Plur|Person=3' in w[5] and 'root' in w[7] and w[1] not in ['есть', 'имеют']:
				kids_rel = get_kids_rel(w, tree)
				text = [w[1] for w in tree] + [w[3] for w in tree]
				if ('около' or 'более' or 'примерно') in text and 'NUM' in text:
					'''
					filter out 84 cases of:
						ежегодно гибнут около 154 тысяч человек
						В настоящее время около суда правопорядок обеспечивают более 500 милиционеров .
						Контролировать ход голосования и подсчет голосов смогут около 2 миллионов наблюдателей .
					'''
					continue
				elif kids_rel.count('nsubj') == 1:
					'''
					Exclude tagging errors where there are two nsunj dependent on plur V-root
					Сейчас все (Plur, 3, nsubj) используют паро - газовый цикл (Sg, 3, nsubj) " , - сказал Чубайс .
					'''
					sg_nsubj_id = choose_kid_by_featrel(w, tree, 'Number=Sing', 'nsubj')
					if sg_nsubj_id:
						'''
						this gets all generic personal sentences:
						на ошибках учатся,
						в компаниях не одобряют
						стадион возводят на новом месте
						во Владикавказе ему готовят радушную встречу
						Демографический кризис в Кузбассе преодолевают не словом , а делом .
						'''
						
						nsubj_kids_pos = get_kids_pos(tree[sg_nsubj_id], tree)
						nsubj_kids_rel = get_kids_rel(tree[sg_nsubj_id], tree)
						if tree[sg_nsubj_id][2] in ['все', 'большинство', 'часть', 'парочка', 'ряд', 'количество',
						                            'половина', 'треть', 'четверть', 'группа']:
							'''
							filter out 60 cases of
								в России сегодня работают достаточное количество банков
								Однако большинство проектов являются
								Почти половина считают , что
							'''
							continue
						elif 'conj' in nsubj_kids_rel:
							'''
							get rid of coordinated sg nsubj which require plural predicate (364 cases)
								Далее идут Лондон , Мадрид и Нью-Йорк .
								Об этом сообщают РИА " Новости " и ИТАР-ТАСС .
								Этого не примут ни общество , ни ученики , ни учителя .
							'''
							continue
						elif 'NUM' in nsubj_kids_pos:
							'''
							discard phrases with паукальные NUM which require N in grammatic sg
								И обе дороги (sg) ведут в бездну бесчеловечности .
								В городе работают три съемочные группы
								Две стороны (sg) Атлантики снова демонстрируют заинтересованность друг в друге .
								Что касается демократов , то три политика (sg) сейчас являются фаворитами .
							'''
							continue
						else:
							
							counter_sem += 1
				
				elif 'nsubj' not in kids_rel:
					'''
					2398
					И пусть меня лучше снимут с выборов , чем Миронов запретит мне сниматься на фоне альма-матер .
					'''
					obj_kid_id = choose_kid_by_featrel(w, tree, 'Number=Plur', 'obj')
					if obj_kid_id:
						if 'Voice=Mid' in w[5]:
							'''
							count in 25 cases
							Уже строятся и так называемые микроспутники
							В эти выходные начнутся работы на квартале Комарова
							в прогулках по вологодским зимним улицам пироги растрясаются быстро
							'''
							counter_sem += 1
						else:
							'''
							filter out 307 typical tagging errors in inverted sents, where true nsubj in postposition is tagged obj
							По всем годам стоят нули , механизма реализации тоже нет
							but not only inverted
							Многие вопросы из повестки московского визита Райс и самарского саммита ЕС-РФ совпадают .
							И пока текут нефтереки с алюминиевыми берегами - все в порядке .
							Слишком уж отдаленными и неосвоенными пока выглядят эти места .
							'''
							continue
					else:
						'''
						what has got thru the sieves above (2066)
							В самом худшем случае его не замечают , не хотят знать .
							Не любят здесь , в Кабарде , нас , чеченцев
							Таким образом здесь решают проблему безработицы .
							К сожалению , меня не слушают .
							В Волгограде , как я понял , гордятся успехами своих милиционеров .
							На постсоветском пространстве Россию обвиняют в неоимперских поползновениях .
						'''
						counter_sem += 1
		
		agentless = allpass - by_pass + counter_sem
	
	return by_pass, agentless


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
					'''
					if not surrounded by spaces gets:
						the best areAS FOR expats for as for
						enTHUSiastic for thus
					'''
					if padded1 in sent or i.capitalize() + ' ' in sent:
						'''
						capitalization changes the count for additive from 1822 to 2182
							(or to 2120 if padded with spaces)
							(furthermore: 1 to 10)
						'''
						res += 1
			except IndexError:
				print('out of range index (blank line in list?: \n', i, file=sys.stderr)
				pass
	return res


def get_epistemic_stance(trees, lang):
	verbs = 0
	
	for tree in trees:
		if lang == 'en':
			'''
			include stance verbs in Tense=Pres with i, we as nsubj into the counts
			'''
			sent = ' '.join(w[1] for w in tree)
			for w in tree:
				if w[1] in ['argue', 'doubt', 'assume', 'believe', 'find'] \
						and has_kid_by_lemlist(w, tree, ['I', 'we']) \
						and has_auxkid_by_tok(w, tree, 'did') == False:
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
		if lang == 'de':
			continue
		
		if lang == 'ru':
			for w in tree:
				if w[2] in ['убежденный', 'уверенный'] and has_kid_by_lemlist(w, tree, ['я', 'мы']) \
						and has_kid_by_lemlist(w, tree, ['быть']) == False:
					verbs += 1
	return verbs


### I have separate counts for all of cconj and sconj
def and_or_counts(trees, lang):
	total = 0
	
	for tree in trees:
		if lang == 'en':
			for w in tree:
				if w[2] in ['and', 'or']:
					total += 1
		if lang == 'de':
			for w in tree:
				if w[2] in ['und', 'oder']:
					total += 1
		if lang == 'ru':
			for w in tree:
				if w[2] in ['и', 'а', 'или', 'либо'] and w[3] == 'CCONJ':
					'''
					the last bit filters out list items used as particles as in:
					Он и в хорошую погоду не ходит гулять
					'''
					total += 1
	return total


def but_counts(trees, lang):
	total = 0
	
	for tree in trees:
		if lang == 'en':
			'''
			mind that next_w = tree[w[0]] which accounts for the difference btw UD indexing and 0-based Py lists
			'''
			for w in tree:
				try:
					if w[2] in ['but'] and tree[w[0]][2] not in ['also']:
						'''
						exclude typical conjunctive (rather than disjunctive) uses in combinations
						She is not only healthy, but also smart.
						'''
						total += 1
				except IndexError:
					continue
		if lang == 'de':
			for w in tree:
				try:
					if w[2] in ['aber'] and tree[w[0]][2] not in ['auch']:
						total += 1
				except IndexError:
					'''
					Überzeugende Belege , die diese Annahme stützen , fehlen aber .
					'''
					continue
		
		if lang == 'ru':
			for w in tree:
				try:
					if w[2] in ['но'] and tree[w[0]][2] not in ['и', 'также']:
						total += 1
				except IndexError:
					'''
					ex. Это могло бы остаться незамеченным , но …
					'''
					continue
	
	return total


## unlike other function this one normalized to the number of adj+adv internally and returns the ratio already!
def comparison_degrees(trees, lang):
	all_ad = 0
	compar = 0
	superl = 0
	'''
	this function averages to the num of ADJ+ADV internally
	We are going to ignore the diff between degrees of adj and adv as nonexistent in DE
	'''
	for tree in trees:
		for w in tree:
			if w[3] == 'ADJ' or w[3] == 'ADV':
				'''
				total counts for ADJ+ADV
				'''
				all_ad += 1
				if 'Degree=Cmp' in w[5]:
					'''
					energy security is more than a matter of assuring short - term supplies
					But we need to modernise sooner rather than later .
					In DE 'spent 45 dollars MORE than last year is not tagged as comp
					'''
					compar += 1
				#                     if compar % 100 == 0:
				#                         print(w[2].upper())
				#                         print(' '.join(w[1] for w in tree))
				if 'Degree=Sup' in w[5]:
					superl += 1
					'''
					Will this be the happiest place in the world
					'''
		for w in tree:
			if lang == 'en' and (w[3] == 'ADJ' or w[3] == 'ADV'):
				'''
				the formants of analytical comparisons are not marked for Degree, but count twds general count
				This explains why the most intense storms are not necessarily the deadliest .
				'''
				comp_kid = choose_kid_by_lempos(w, tree, 'more', 'ADV')
				sup_kid = choose_kid_by_lempos(w, tree, 'most', 'ADV')
				if comp_kid:
					compar += 1
					'''
					You become more productive and do better work as a result .
					'''
				if sup_kid:
					superl += 1
					'''
					They are also one of the most destructive creatures on the planet .
					'''
			
			if lang == 'de':
				if w[2] == 'mehr' and not 'Degree=Cmp' in w[5]:
					mehrs_head = get_headwd(w, tree)
					'''
					Wir stimmen beispielsweise mit denjenigen überein , die mehr Transparenz in der WTO fordern
					'''
					if mehrs_head and mehrs_head[3] == 'VERB':
						compar += 1
			
			if lang == 'ru' and (w[3] == 'ADJ' or w[3] == 'ADV'):
				if w[2] == 'больший' and 'Degree=Pos' in w[5]:
					compar += 1
					'''
					еще в большей мере
					все большую популярность
					по большей части вывоз нефти и газа
					'''
				if w[2].startswith('наи') and 'Degree=Pos' in w[5] and not w[2] in ['наивный', 'наискосок']:
					superl += 1
					'''
					Наибольшее оживление в зале вызвала дискуссия о
					На нем были обозначены наиболее острые проблемы в данной сфере
					наилучший способ избежать
					'''
				
				sup_kid0 = choose_kid_by_lempos(w, tree, 'наиболее', 'ADV')
				sup_kid1 = choose_kid_by_lempos(w, tree, 'самый', 'ADJ')
				if sup_kid0:
					superl += 1
				if sup_kid1:
					superl += 1
					'''
					А самым серьезным доказательством победы искусства СССР стала
					'''
	compar = compar / all_ad
	superl = superl / all_ad
	
	return compar, superl


def polarity(trees, lang):
	negs = 0
	for tree in trees:
		for w in tree:
			if lang == 'en':
				if w[2] in ['no', 'not', 'neither']:
					negs += 1
					'''
					The UK is n't some offshore tax paradise .
					America no longer has a Greatest Generation .
					But almost no major economy scores in the top 10
					'''
			if lang == 'de':
				if w[2] in ['kein', 'nicht']:
					negs += 1
					'''
					Aber es gibt wohl keinen Patienten , der gegen ..
					In diesem Fall wirkt das Kalzium allerdings nicht elektrisch , sondern chemisch .
					'''
			if lang == 'ru':
				if w[2] in ['нет', 'не']:
					negs += 1
					'''
					которых ни у каких претендентов на власть , как правило , нет
					Никаких сенсаций не будет , не рассчитывайте " , - сказал он журналистам .
					'''
	return negs


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


def demdeterm(trees, lang):
	res = 0
	for tree in trees:
		for w in tree:
			if lang == 'en':
				'''
				the list is ranked by frequency
				'''
				if w[7] == 'det' and w[2] in ['this', 'some', 'these', 'that', 'any', 'all',
				                              'every', 'another', 'each', 'those',
				                              'either', 'such']:
					res += 1
			
			if lang == 'de':
				if w[7] == 'det' and w[2] in ['dies', 'alle', 'jed', 'einige', 'solch', 'viel',
				                              'ander ', 'jen', 'all', 'irgendwelch',
				                              'dieselbe', 'jeglich', 'daßelbe', 'irgendein', 'diejenigen']:
					res += 1
			
			if lang == 'ru':
				'''
				for Russian there is no distinction between эти полномочия и его полномочия
				listed here in order of freq
				'''
				## muted item: 'свой',
				if w[7] == 'det' and w[2] in ['этот', 'весь', 'тот', 'такой', 'какой',
				                              'каждый', 'любой', 'некоторый', 'какой-то',
				                              'один', 'сей', 'это', 'всякий', 'некий', 'какой-либо',
				                              'какой-нибудь', 'кое-какой']:
					res += 1
	return res


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


### 7 UD dependencies
def relation_distribution(tree, i_relations):
	sent_relations = [w[7] for w in tree]
	distribution = {rel: sent_relations.count(rel) for rel in i_relations}
	'''
	count method counts acl:relcl for acl, which is a desired behavior
	'''
	
	# Converting to probability distribution
	# 'probabilities' are basically ratio of the rel in question to all rels in the sentence
	total = sum(distribution.values())
	if total:  # Only if there are from our limited list of UD relations in the sentence
		for key in distribution:
			distribution[key] /= total
	return distribution


def ud_probabilities(trees, lang):
	relations = "acl aux aux:pass ccomp nsubj:pass parataxis xcomp".split() # mark
	'''
	previous research has shown that
		aux, ccomp, acl:relcl, mark, xcomp,
		parataxis and nsubj:pass; aux:pass
	are good indicators of translationese
	as German does not have acl:relcl I am falling back to the more general type of relation
	'''
	relations_all = 'nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse aux cop ' \
	                'mark nmod appos nummod acl amod det clf case conj cc fixed flat compound list parataxis orphan ' \
	                'goeswith reparandum punct root dep acl:relcl flat:name nsubj:pass nummod:gov aux:pass ' \
	                'flat:foreign obl:agent nummod:entity'.split()
	relations_d = {rel: [] for rel in relations}
	for tree in trees:
		rel_distribution = relation_distribution(tree, relations_d)
		'''
		this returns probabilities for the picked dependencies in this sent
		'''
		for rel in relations_d.keys():  # reusing the empty dict
			'''
			this collects and average the probability stats over sents of the text to the global empty dict
			'''
			relations_d[rel].append(rel_distribution[rel])
			'''
			values in this dict are lists of probabilities in each sentence; need to average them
			'''
	dict_out = {}
	for rel in relations_d.keys():
		dict_out[rel] = np.average(relations_d[rel])
	
	return dict_out
