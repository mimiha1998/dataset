UPD: the project is archived 3 June 2022
see the submitted version of the thesis (and a paper based on it) in submissions/ folder. 

@inproceedings{Poltorak2022,
	location = {Rhodes Island, Greece},
	author = {Poltorak, Kateryna and Kunilovskaya, Maria},
	booktitle = {Proceedings of the New Trends in Translation and Technology},
	editor = {},
	month = {4--6 July},
	pages = {82--90},
	publisher = {},
	title = {{Computational Approaches to Register as a Factor in English-to-Spanish Translation}},
	year = {2022},
	url = {}
}

This project is largely an adaptation of approach used on EN-RU (see paper below) to EN-ES.

@incollection{Kunilovskaya2021regs,
	author = {Kunilovskaya, Maria and Corpas Pastor, Gloria},
	booktitle = {New Frontiers in Translation Studies},
	doi = {10.1007/978-981-16-4918-9\_6},
	editor = {Wang, Vincent X. and Li, Defeng and Lim, Lily},
	isbn = {978-981-16-4917-2},
	issn = {21978697},
	pages = {133--180},
	address = {Singapore},
	publisher = {Springer Nature},
	title = {{Translationese and register variation in English-to-Russian professional translation}},
	year = {2021}
}

### Register variation in English-to-Spanish translation

Main RQ: How different is translated Spanish across registers from comparable non-translated language

H1: The amount of translationese is contingent on the distance between source and target language in a given register.

H2: Each register produces its own translations indicators.

+ What are register specific translationese indicators 

2 May 2022: Download the input table for univariate analysis ([booked_debates_fiction1.tsv](https://drive.google.com/file/d/1gWOmHhV7Vn36sT_k-UBeX9LWZ8sQy5rv/view?usp=sharing)) with 3_univariate_analyses.py and put it in the data/ folder.

```bash 

Spoiler (on 28, not on 60 feats):
+ debates: 'mdd', 'content_dens', 'content_TTR', 'ccomp', 'parataxis'
+ fiction: 'nn', 'content_TTR', 'simple', 'nnargs', 'parataxis'
```

+ Can they be theoretically motivated?

+ Can neural approaches (distributed representations from Sentence Encoders) return similar classification accuracy? 

### Data Collection

**TASK:** follow the selection criteria and collect parallel data for EN > ES in a number of registers

Criteria for selecting sources of data
- Available for download
- Clear register mark-up and register diversity 
- Translation direction English-to-Spanish
- Refence corpora availability
- Any level of alignment (doc/sent), preferably both
- Which bitext format is used (XML, TMX, tsv)

#### Corpus structure
```bash
kateryna/corpus$ tree
├── parsed
     ├── debates
         ├── ref
             └── es
                (1031 files)
                ├── 19990721.ES.conllu
                ├── 19990722.ES.conllu
         ├── src
             └── en
                (539 files)
                ├── 19990720.EN.conllu
                ├── 19990721.EN.conllu
        └── tgt
            └── es
                (539 files)
                ├── 19990720.ES.conllu
                ├─/home/u2─ 19990721.ES.conllu
                
     └── fiction
        ├── ref
            └── es
                (257 files)
                ├── ref_10_chunk_11.conllu
                ├── ref_10_chunk_12.conllu
         ├── src
             └── en
                (144 files)
                ├── src_1_chunk_10.conllu
                ├── src_1_chunk_11.conllu
        └── tgt
            └── es
                (144 files)
                ├── Resulttgt_1_chunk_10.conllu
                ├── tgt_1_chunk_11.conllu
                
└── txt
    ├── debates/home/u2
        ├── ref
            └── es
                ├── 19990721.ES.txt
                ├── 19990722.ES.txt
        ├── src
            └── enRFE
                ├── 19990720.EN.txt
                ├── 19990721.EN.txt
        └── tgt
            └── es
                ├── 19990720.ES.txt
                ├── 19990721.ES.txt

    └── fictiGeneral data shape:onResult
          ├── ref
              └── es
                  ref_10_chunk_11.txt
                  ref_10_chunk_12.txt
          ├── src
              └── en
                  ├── src_1_chunk_10.txt
                  ├── src_1_chunk_11.txt
  RQ2: 
          └── tgt
              └── es
                  ├── tgt_1_chunk_10.txt
                  ├── tgt_1_chunk_11.txt
```

**Outcome:**

src, tgt and ref in two registers: fiction and debates 
(see code which selects and cleans textual data from existing corpora in get_texts/)

## Data preprocessing
To populate the corpus/ folder unpack corpus.zip from 
[download](https://drive.google.com/file/d/1VXsuoYF2aPTtdlx6hJd9lvcDMdO5qrCt/view?usp=sharing). This is the version of the corpus preprocessed as described below (as of 18 January 2022)

**UPDATE REQUIRED: As of Jan 26, all statistics need updating after Kateryna fixes the faults with corpus formating (primarily bad line breaks)**

**Details on the raw textual data (all subcorpora are in one-sentence-per-line format):** 
NB! very short texts were deleted (files with less than 10 sentences and less than 400 tokens, in parallel corpus the counts are based on translation):

(see respective lists of removed files in [help_lists](https://github.com/kunilovskaya/kateryna/blob/master/help_lists/))

debates:RQ2: 
* ref: 28 files 
* src-tgt: 17 doc pairs, inc. 20070115.EN/ES with badly mismatching 8/24 sentences respectively

fiction:/home/u2
* ref: 14 files (e.g. 'ref_10_chunk_1.txt', 'ref_3_chunk_1.txt', 'ref_9_chunk_1.txt')
* src-tgt: 0 doc pairs (this parallel corpus seems to have no short docs)

Table 1 Statistics on the resulting corpus of 2654 txt files (done with ~/proj/kateryna$ python3 wc_walks_txt_folders.py):

| subcorpus |   Words |  Sents | Texts |
|----------:|--------:|-------:|-------|
|   debates |         |        |       |
|       ref | 5654116 | 183117 | 1031  |
|       src |     ??? |    ??? | 539   |
|       tgt |     ??? |    ??? | 539   |
|   fiction |         |        |       |
|       ref |     ??? |    ??? | 257   |
|       src |  542834 |  27090 | 144   |
|       tgt |  502697 |  31380 | 144   |

Table 2 Homogeneity of the subcorpora (mean wc and STD):

| subcorpus |           Av. Words | Av. Sents         |
|----------:|--------------------:|-------------------|
|   debates |                /home/u2     |                   |
|       ref |   5484.1 (+/- 7648) | 177.6 (+/- 239)   |
|       src | 6231.5 (+/- 3312.5) | 270.3 (+/- 145.1) |
|       tgt | 6501.5 (+/- 3792.8) | 262.5 (+/- 154.7) |
|   fiction |                     |                   |
|       ref | 3139.3 (+/- 4958.7) | 199 (+/- 312.4)   |
|       src | 3769.7 (+/- 2221.5) | 188.1 (+/- 152.8) |
|       tgt | 3491.0 (+/- 195/home/u21.1) | 217.9 (+/- 152.5) |


Did you do any of the following on the texts before putting them through parsing or at the parsing time? 

- document selection (discarding any documents from the existing corpus; on which criteria?)
- html-tag removal/ noise reduction (deleting parts of documents)
- filtering out short sentences (yes, we ignore one-word sentences, see 'minlen' keyworded argument)
```bash
see, for example, a sentence from ref_9_chunk_9.conllu
# sent_id = 419
# text = —
1	—	-	PUNCT	.	_	0	root	_	_
```
- spelling normalisation (bringing the variety of possible double quotes to the standard straight quote: ")

UD parsing with english-ewt-ud-2.5-191206.udpipe and spanish-gsd-ud-2.5-191206.udpipe

**Details on the parsed texts:** 
u2@MAK:~/proj/kateryna$ python3 wc_walks_UDfolders.py --root corpus/parsed --minlen 2 --depth 3

Table 3 

| subcorpus |           Words |  Sents | Texts   |
|----------:|----------------:|-------:|---------|
|   debates |                 |        |         |
|       ref |         6335046 | 189526 | 1031    |
|       src |     ~~3844826~~ | ~~152638~~ | ~~540~~ |
|       tgt |     ~~3914618~~ | ~~144853~~ | ~~540~~ |
|   fiction |                 |        |         |
|       ref |      ~~944489~~ |  ~~53198~~ | ~~258~~ |
|       src |          654242 |  30094 | 144     |
|       tgt |          587734 |  33365 | 144     |

Ignored sentence splitting errors (ex. "3.", "II."): 40337
Short sentences (< 2): 716

Notes: ref_es from [RNC parallel ES > RU subcorpus](https://ruscorpora.ru/new/search-para-es.html)

RQ2: Can they be theoretically motivated
6-13 December tasks:
- save the parallel structural parts of translated novels (src and tgt) such as chapters as separate files to get a greater number of observations;
we cannot use chunking (by 100/home/u2 consecutive sentences) because we are likely to lose the parallelism of the src and tgt chunks.
- extract the basic set of translationese features
- exhaustive description of the sources of textual data 
  (content in detail: register, time of production, whether authors are all unique (representativeness); balance in terms of document size)

## Feature engineering and extraction

28 Jan 2022 - a programme extracting 60 features based on UD annotation and pre-defined lists (new_mega_collector.py+helpfunctions.py+extractors.py) is ready.
They include: 
* 29 features are frequencies of the respective universal syntactic tags, averaged across all sentences in each document
* 31 features reflecting frequencies of PoS and syntactic tags and their combinations; positive list-based filters are used to reduce noise

This is the full list (see a description [here](https://github.com/kunilovskaya/kateryna/blob/master/get_feats/19_indep_features_description.odt)):

```bash
'wc', 'sents', 'sentlength', 'wdlength', 'interrog', 'nn', 'mhd', 'mdd', 'content_dens', 'content_TTR', 'finites', 
'attrib', 'pasttense', 'addit', 'advers', 'caus', 'tempseq', 'epist', 'numcls', 'simple', 'nnargs', 'ppron', 
'possp', 'intonep', 'conj', 'sconj', 'neg', 'copula', 'determ', 'propn', 'adp', 'acl', 'advcl', 'advmod', 'amod', 
'appos', 'aux:pass', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj.1', 'cop', 'csubj', 'dep', 'det', 'discourse', 
'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 
'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp'

```
Metadata columns include: 'doc', 'register', 'type', 'lang', 'book_id'.

**General data shape:**

- Number of observations (rows): 2654 
- Number of input features (columns, excluding meta): 60


## Methods
### Text classification experiment setup
While in the fiction subcorpus many chunks come from a few literary works (10 in reference subcorpus and 7 in the parallel subcorpus, (e.g. ref_1, src/tgt_1)), 
we want to ensure that the ML model (SVM) does not overfit and generalizes well to the unseen books. 
To this end, we used scikit-learn GroupKfold algorithm which generates crossvalidation folds where training and testing sets do not include chunks from the same books. 
The books (i.e. groups) to be included in train and test are drawn equally from each class.
(also see a description of a similar approach in [Translationese in Russian Literary Texts](https://aclanthology.org/2021.latechclfl-1.12/))

For example, 
* in the language contrast classification (total of 37 groups, en_src vs es_ref), we train on 32 books and test on the chunks from the 5 unseen books in each of the 7 folds
* intra-linguistic register contrast: 
   1. en_src_debates vs en_src_fiction - 17 groups - 15 in train, 2 in test
   2. es_ref_debates vs es_ref_fiction - 20 groups - 17 in train, 3 in test


NB! For consistency reasons, 
* we randomly grouped the texts from the debates subcorpora (ref and parallel) into 30 artificial "books" (e.g. ref_01, src/tgt_01).
* in all experiments we usRQ2: Can they be theoretically motivatede 7-fold cross-validation
* the chance-level baseline is implemented using sklearn DummyClassifier(strategy='stratified')

