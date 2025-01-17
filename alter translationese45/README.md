## Translationese feature extraction for English, German and Russian

Python3 code to extract 45 translationese indicators for English, German and Russian. 
Most of them were used in the research presented at LREC 2020

Related publication:
- *Kunilovskaya, Maria & Ekaterina Lapshinova-Koltunski (2020). Lexicogrammatic translationese across two targets and competence levels. 
LREC-2020, Marseille, May 11-15, 2020.*
(link and bib to be added)

In developing the extraction procedures and the related pre-defined lists of items, we roughly follow the suggestions by
* for English: Nini, A. 2015. Multidimensional Analysis Tagger (Version 1.3) based on Biber, D. (1988). Variation across speech and writing (2nd ed.). Cambridge University Press, 
* for German: Evert, S. and Neumann, S. (2017). The impact of translation direction on characteristics of translated texts : A
multivariate analysis for English and German. Empirical Translation Studies: New Methodological and Theoreti-
cal Traditions, 300:47. based on Neumann, S. (2013). Contrastive register variation. A quantitative approach to the comparison of English and German. Mouton de Gruyter, Berlin, Boston.
* for Russian: Katinskaya, A. and Sharoff, S. (2015). Applying Multi-Dimensional Analysis to a Russian Webcorpus: Searching for Evidence of Genres. The 5th Workshop on Balto-Slavic Natural Language Processing, pages 65–74, September.

The detailed description of the features is provided in *lrec20_45featureset_description.pdf*.

To reproduce the extraction of frequencies for the 45 translationese features for the three languages:
- clone the repository to your user's home directory (/home/username/);
- unpack the data in the preprocessed folder and delete the archives; 
- change the username in rootdir option in the mega_collector.py;
- adjust the output file name in mega_collector.py, if necessary. By default the output (out.tsv) is created in the project rootdir;
- change the username in lists_path in helpfunctions.py module. 

The archives contain trees of folders for each language pair, including professional and student translations 
with their sources as well as the non-translated reference texts used. The folder names are used as class labels by the script.
Each file contains a preprocessed and UD parsed text in the *.conllu format from the respective subcorpus. 
Note: our subset of the Croco corpus is available on request. 

- Install the necessary dependencies, specifically python igraph-python 0.7.4 library required to extract mean hierarchical distance

``
sudo apt-get install python3-igraph
``
- run
```
python3 mega_collector.py
```
The spreadsheet *our45features_extracted.tsv* has our output and is added for reference.





