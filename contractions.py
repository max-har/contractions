#!/usr/bin/env python
# coding: utf-8

# # Contractions and Ambiguity
# ## Grammar Based Disambiguation of English Apostrophe+S Contractions in Movie Scripts
# ##### Max Harder, 2919411, max.harder@uni-bielefeld.de
# A mini project (10 h) for _230028 Natural language processing:\\Introduction to text analysis (S) (SoSe 2019)_, Mr Nikolai Ilinykh.

# In[1]:


# treetaggerwrapper: "FutureWarning: Possible nested set at position […]"
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import treetaggerwrapper
import nltk
import re
import glob
import spacy # POS tagging, NER
import numpy as np # scientific computing
import matplotlib.pyplot as plt
import os

from pprint import pprint # vgl. https://docs.python.org/3/library/pprint.html
from nltk import sent_tokenize
from collections import Counter # dict subclass for counting hashable objects
from tqdm import tqdm # progress bar


# In[3]:


base_path = './data/'
output_path = './output/'
file_path = glob.glob(base_path + 'quentin-tarantino_*.txt')
file_path_PREPROC = glob.glob(base_path + 'PREPROC_*.txt')


# ### Preprocessing

# ##### Make Preprocessed File

# In[4]:


def contractions_in_paragraph(this_par_list, all_par_list):
    """
    extract preprocessed sentences that cointain contraction(s)
    param1: list of strings of current paragraph
    param2: list of all paragraphs
    output: appended list of all paragraphs
    """
    joined_par = " ".join(this_par_list)
    # lower and normalize apostrophes
    joined_par = re.sub(r"[“”]", "\"", joined_par.lower().replace("’", "\'"))
    joined_par = re.sub(r"[\(\)\[\]\{\}\*]", "", joined_par)
    if re.search("\'s", joined_par):
        t_sents = sent_tokenize(joined_par)
        # extract sentence that contains contraction(s)
        for sent in t_sents:
            if re.search("\'s", sent):
                all_par_list.append(sent)
    return all_par_list


# In[5]:


def make_preproc_file(path_to_files, preproc_file_name):
    """
    check each paragraph for contractions and make file with contraction sentences.
    param1: path to files
    param2: name of preproc file
    output: None; makes file    
    """
    # make empty file
    with open(preproc_file_name, 'w', encoding='utf-8') as f:
            pass
    # iterate files
    for file in tqdm(path_to_files):
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        this_par = []
        all_par = []
        for i, line in enumerate(data):
            # avoid indexError when last line is reached
            try: 
                # remove speakers and instructions
                if re.match(r"[A-Z0-9\.\-\(\) ]+$", line):
                    pass
                # continuation in next line
                elif re.match(r"." , line) and re.match(r"." , data[i+1]):
                    this_par.append(line.replace("\n", "").strip())
                # next line empty
                elif re.match(r"." , line) and not re.match(r"." , data[i+1]):
                    this_par.append(line.replace("\n", "").strip())
                # empty line
                else:
                    all_par = contractions_in_paragraph(this_par, all_par)
                    this_par = []
            # last line belongs to paragraph (cannot check next line)
            except IndexError:
                this_par.append(line.replace("\n", "").strip())
                all_par = contractions_in_paragraph(this_par, all_par)
        # join all paragraphs
        joined_all_par = "\n".join(all_par)
        # append to preproc file
        with open(preproc_file_name, 'a', encoding='utf-8') as f:
            # write file tag
            f.write("<meta file="+file[7:]+">\n")
            # write contraction sentence
            f.write(joined_all_par)
            f.write("\n")
    return None


# In[6]:


make_preproc_file(file_path, "./data/PREPROC_scripts.txt")


# ##### Read Preprocessed File

# In[7]:


def read_preproc_file(file_path):
    """
    read preprocessed file and make list with data
    param1: file path to file with preprocessed contraction sentences
    output: list with contraction sentences of each file in sub-lists
    """
    this_file = []
    all_files = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not re.match("^<meta file=[\w_-]+\.txt>$", line):
                this_file.append(line)
            else:
                all_files.append(this_file)
                this_file = []
        all_files.append(this_file)
        # delete first empty element
        del all_files[0]
    return all_files


# In[8]:


# length is number of preprocessed files
all_files = read_preproc_file("./data/PREPROC_scripts.txt")
len(all_files)


# In[9]:


# number of sentences with contractions for each file
[(index, len(element)) for index, element in enumerate(all_files)]


# ### Processing

# ##### Part-of-Speech-Tagging

# Used tag set: http://www.laurenceanthony.net/software/tagant/resources/treetagger_tagset.pdf

# In[10]:


def make_tags(all_files_list):
    """
    POS tag all contraction sentences of each file
    param1: list with contraction sentences of each file in sub-lists
    output: list of all tagged files in sub-lists    
    """
    all_tagged_files = []
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
    for each_file in tqdm(all_files_list):
        tagged_sentences = []
        #each_file = each_file[:10] # SUBSET; uncomment for development
        for sent in each_file:
            tags = tagger.tag_text(sent)
            made_tags = treetaggerwrapper.make_tags(tags)
            tagged_sentences.append(made_tags)
        all_tagged_files.append(tagged_sentences)
    return all_tagged_files


# In[11]:


all_tagged_files = make_tags(all_files)
#all_tagged_files


# In[12]:


#len(all_tagged_files[8])


# ##### Phrase Chunking (Noun Phrases)

# In[13]:


def extract_noun_chunks(sentence):
    '''
    extract noun chunks of sentence
    param1: sentence (string)
    output: list of tuples with start position and noun chunk
    '''
    nps = [np.text for np in nlp(sentence).noun_chunks]
    # get start positions of each noun chunk
    # workaround: empty dummy list if same noun chunk occurrs more than once
    start = [re.search(str(element), sentence).start() for element in nps if len(re.findall(str(element), sentence)) == 1]
    i_nps = zip(start, nps)
    return list(i_nps)


# In[14]:


# example
#extract_noun_chunks("this a is a huge car and that is a super huge car.")


# ##### Named Entity Recognition

# In[15]:


def named_entities(sentence):
    """extract named entities of sentence into list"""
    doc = nlp(sentence)
    nes = [ent.text for ent in doc.ents]
    return nes


# ##### Expanding Contractions

# In[16]:


# load spacy
nlp = spacy.load('en_core_web_sm')

def classify_contractions(t_sentence):
    """
    takes tagged sentence with contraction and classifies contraction(s)
    param1: tagged sentence with contraction (list)
    output: type of contraction
    """
    result = None
    sent = " ".join([tag[0] for tag in t_sentence])  # make sentence
    all_conts = []
    enum_noun_chunks = extract_noun_chunks(sent)  # enumerated
    # iterate words in sentence
    for i, t_word in enumerate(t_sentence):
        # check for each contraction
        if re.search(r"^\'s$", t_word[0]) and i != len(t_sentence)-1:

            # 1. the possesive marker
            # in the beginning of a noun phrase in the article position
            if (re.search(r"N(N|P)(S)?", t_sentence[i-1][1]) and
                (re.search(r"N(N|P)(S)?", t_sentence[i+1][1]) or
                 re.search(r"JJ(R|S)?", t_sentence[i+1][1]) or
                 re.search(r"SENT|,|:", t_sentence[i+1][1]) or
                 re.search(r"CD", t_sentence[i+1][1]))):
                result = ("'s", "POS")
            # does not use noun chunks (sometimes removes ’s)

            # 2. an abbreviation for the copula 'is' (or 'was')
            # distinction between present tense and past tense ignored
            # in front of an adjective or adverb
            elif (re.search(r"JJ(R|S)?|RB|DT", t_sentence[i+1][1]) and not   # DT might conflict with 'has'
                  re.search(r"not", t_sentence[i+1][0])):
                result = ("is/was", "VBZ (copula)")
            elif (len(t_sentence)-i > 2 and
                  re.search(r"not", t_sentence[i+1][0]) and
                  re.search(r"JJ(R|S)?|RB|DT", t_sentence[i+2][1])):  # DT might conflict with 'has'
                result = ("is/was", "VBZ (copula)")
            # in front of a noun phrase
            if result == None:  # dummy elif
                for noun_chunk in enum_noun_chunks:
                    k = re.compile(r'\b{}\b'.format(t_sentence[i+1][0]), re.I)
                    if re.search(k, noun_chunk[1]) and re.search(t_word[0], sent).start() < noun_chunk[0]:
                        result = ("is/was", "VBZ (copula)")

            # 3. an abbreviation for the auxiliary 'is' (or 'was')
            # distinction between present tense and past tense ignored                  
            # following verb in present participle form
            elif re.search(r"VVG", t_sentence[i+1][1]):
                 result = ("is/was", "VBZ (auxiliary)")
            # perhaps a ‘not’ or an adverb intervening.
            elif (len(t_sentence)-i > 2 and
                  re.search(r"RB", t_sentence[i+1][1]) and
                  re.search(r"VVG", t_sentence[i+2][1])):
                result = ("is/was", "VBZ (auxiliary)")

            # 4. an abbreviation for the auxiliary 'has'
            # in front of a past participle verb form
            elif re.search(r"V(B|H|V)N", t_sentence[i+1][1]):
                  result = ("has", "VHZ")
            # perhaps a ‘not’ or an adverb intervening.
            elif (len(t_sentence)-i > 2 and
                  re.search(r"RB", t_sentence[i+1][1]) and
                  re.search(r"V(B|H|V)N", t_sentence[i+2][1])):
                result = ("has", "VHZ")

            # 5. an abbreviation for the auxiliary 'does'
            # question and a verb that is neither present participle nor past participle
            elif (len(t_sentence)-i > 2 and
                  t_sentence[-1][0] == "?" and
                  re.search(r"PP", t_sentence[i+1][1]) and
                  (re.search(r"^V(B|D|H|V)$", t_sentence[i+2][1]) or
                   re.search(r"got", t_sentence[i+2][0]))):
                result = ("does", "VDZ")
                    
            # 6. an abbreviation for the pronoun 'us'
            elif t_sentence[i-1][0] == "let":
                result = ("us", "PP (us)")
                #print(result, t_sentence) # now catches something

            # 7. the plural marker for abbreviations, acronyms and numbers.
            # follow a number [or plural indicator like all or many],
            # or if it occurs at the end of the sentence.
            elif (re.search(r"CD", t_sentence[i-1][1]) and
                  re.search(r"SENT", t_sentence[i+1][1])):
                result = ("'s", "N/A (plural marker)")
            elif (len(t_sentence)-i < len(t_sentence)-1 and
                  re.search(r"all|many", t_sentence[i-2][0]) and
                  re.search(r"CD", t_sentence[i-2][1])):
                result = ("'s", "N/A (plural marker)")

            # not identified
            if result == None:
                tags_words = " ".join([t_sentence[i-1][1], t_sentence[i][1], t_sentence[i+1][1],
                                t_sentence[i-1][0], t_sentence[i][0], t_sentence[i+1][0]])
                #print(tags_words)
                return (False, sent, tags_words)
           
            # collect all contractions of each sentence
            all_conts.append(result)
        
        # else: contraction is end of sentence
        elif re.search(r"\'s$", t_word[0]) and i == len(t_sentence)-1:
            # filter: contraction follows named entity (number, name)
            if t_sentence[i-1][0] in named_entities(sent):
                # follows number: plural marker
                if t_sentence[i-1][1] == "CD":
                    result = "N/A (plural marker)"
                # follows name: possesive marker
                else:
                    result = "POS"
            # remains unidentified
            else:
                hit = " ".join([t_sentence[i-1][1], t_sentence[i][1], t_sentence[i-1][0], t_sentence[i][0]])
                #print(hit)
                return (False, sent, hit)
            # collect contractions identified in this filter
            all_conts.append(result)

    #print(sent, all_conts)
    return sent, all_conts


# In[17]:


def replace_contractions(this_cont_result):
    """
    param1: list of all contraction results
    """
    sent, tag_list = this_cont_result
    replaced = sent.replace("\'s", "{}", 1)
    for i in range(len(tag_list)):
        sent = replaced.format(tag_list[i])
        replaced = re.sub("^\'s$", "{}", sent, 1)
    return this_cont_result[0], replaced


# In[18]:


def make_output_file(output_file_name, all_tagged_files):
    """
    make output file
    param1: name of output file
    param2: list of all tagged files in sub-lists
    output: list of unclassified contractions
    """    
    # make empty file
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(output_path+output_file_name, 'w', encoding='utf-8') as f:
        pass
    left_over = []  # for unindentified contractions
    with open(output_path+output_file_name, 'a', encoding='utf-8') as f:  
        for index, file in tqdm(enumerate(all_tagged_files)):
            # indicate current file (number) in output file
            f.write("<meta number="+str(index+1)+">\n")
            for t_sentence in file:
                cont_result = classify_contractions(t_sentence)
                if not cont_result[0] == False:
                    sent, replaced = replace_contractions(cont_result)
                    f.write(replaced+"\n")
                # unidentified contracions
                else:
                    left_over.append(cont_result)
    return left_over


# In[19]:


left_over = make_output_file("OUTPUT_scripts.txt", all_tagged_files)
#left_over


# In[20]:


def make_missing_file(missing_file_name, missing_list):
    """
    make file for unidentified contractions
    param1: name of file for unidentified contractions
    param2: list of tuples for unidentified contractions
    output: None
    """
    with open(output_path+missing_file_name, 'w', encoding='utf-8') as f:
        for entry in missing_list:
            f.write(entry[2]+"\t"+entry[1]+"\n")


# In[21]:


make_missing_file("MISSING_scripts.txt", left_over)


# ----------------------------
