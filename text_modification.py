# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:50:26 2017

@author: anilm
"""
import os, pickle, threading, re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import pandas as pd
import numpy as np

# Constants we want to use
filename = 'english_corpus/google-10000-english.txt'
__ENGLISH_DICTIONARY__ = set(line.strip() for line in open(filename))
pickles_folder = "pickles"
if not os.path.exists(pickles_folder):
    os.makedirs(pickles_folder)
pickle_extension=".pickle"
DATA_DIR = 'data'
vocabulary = dict()
window_size =2

def check_eligibility(word):
    """
    This method checks if a word is eligible to be chosen as a useful word. For a word to be chosen the following are the conditions
        1. The has to be alpha-numeric
        2. length of the word is more than 1
        3. The words is not a number(digit)
        4. The word is not in the english dictionary from the 'Google 10000 words list'
    """
    if not re.match('^\d*[a-zA-Z][a-zA-Z0-9]*$', word):
        return False
    if len(word) <= 1:
        return False
    if word.isdigit():
        return False
    if word in __ENGLISH_DICTIONARY__:
        return False
    return True

def clean_up_a_little(words):
    """
    This method takes a list of words and then returns only useful words. First, each word is split further if there is one of
    [['/'],['-'],['.'],[',']] these special characters in the word. Then useful words are selected if they return true from the
    check eligibility method
    """
    spl_char_splitter = re.compile('[\/|\-|\.|,]')
    useful_words = []
    stemmer = PorterStemmer()
    for w in words:
        w = w.lower()
        if re.search(spl_char_splitter, w):
            ws = re.split(spl_char_splitter, w)
            if check_eligibility(ws[0]):
                useful_words.append(stemmer.stem(ws[0]))
            if check_eligibility(ws[1]):
                useful_words.append(stemmer.stem(ws[1]))
        else:
            if check_eligibility(w):
                useful_words.append(stemmer.stem(w))
    return useful_words

def build_phrases_vocabulary(words):
    res = []
    for i in range(0, len(words) - window_size +1):
        for j in range(1, window_size+1):
            phrase = words[i]
            if (i+j) < len(words):
                for x in range(1, j+1):
                    phrase = phrase +" "+words[i+x]
            else:
                break
            res.append(phrase)
    return res

def modify_text(df_chunk, vocab_flag):
    """
    This method works on a single chunk of the original dataframe. Multiple threads run this method to modify the whole input.
    Text portion of each row is first tokenized into sentences and then words. The words are cleaned up and only useful words
    are returned. If the vocab_flag is True, then vocabulary is also generated using these words. Once the useful words are
    selected then the sentence is reconstructed and the whole text is replaced in the dataframe.
    """
    lim = df_chunk.shape[0]
    for i in range(0, lim):
        modified_text = ""
        target_text = df_chunk.iloc[i,df_chunk.columns.get_loc('Text')]
        sents = sent_tokenize(target_text)
        for sent in sents:
            words = word_tokenize(sent)
            useful_words = clean_up_a_little(words)
            phrases = build_phrases_vocabulary(useful_words)
            if vocab_flag:
                for phr in phrases:
                    vocab_dump[phr] += 1
            modified_text += " ".join(useful_words)
            modified_text += ". "
        df_chunk.iloc[i,df_chunk.columns.get_loc('Text')] = modified_text
    filename = os.path.join(pickles_folder,threading.currentThread().getName()+pickle_extension)
    df_output = open(filename, 'wb')
    pickle.dump(df_chunk, df_output)
    df_output.close()

def make_sure_pickles_folder_exists_and_empty():
    """
    Different threads dump their chunks of modified dataframe into a pickles folder and then merge them to a single dataframe.
    This method makes sure that the folder is empty before we begin new run.
    """
    if os.path.exists(pickles_folder):
        for our_file in os.listdir(pickles_folder):
            file_path = os.path.join(pickles_folder, our_file)
            os.unlink(file_path)
    else:
        os.makedirs(pickles_folder)
            
def generate_modified_datafile(input_dataframe, num_threads, vocab_flag):
    """
    In the case that the modified file is not in the data directory, a modified dataframe is generated from input data.
    Also parts of input data are run separately based on the num of threads specified.
    """
    thread_limit = int(np.ceil(input_dataframe.shape[0]/num_threads))
    allThreads = []
    make_sure_pickles_folder_exists_and_empty()
    for tix in range(0, num_threads):
        s = tix*thread_limit
        e = (tix+1)*thread_limit
        df_chunk = input_dataframe.iloc[s:e,:]
        thread_name = "thread_"+str(tix)
        thread = threading.Thread(name=thread_name, target=modify_text, args=[df_chunk, vocab_flag])
        allThreads.append(thread)
        thread.start()
    for t in allThreads:
        t.join()
    modified_df = pd.DataFrame()
    for tix in range(0, num_threads):
        pickle_filename = os.path.join(pickles_folder, "thread_"+str(tix) + pickle_extension)
        output = open(pickle_filename, 'rb')
        df = pickle.load(output)
        output.close()
        modified_df = pd.concat([modified_df, df], axis=0)
    
    modified_df.drop_duplicates('Text', inplace=True)
    return modified_df

def fetch_modified_dataframe(input_dataframe, train_flag=False, num_threads=4):
    """
    This method will modify the input dataframe and also constructs the vocabulary if the train flag is True.
    Method will first create filename based on train_flag. If both the modified file and vocabulary exists in the data 
    directory, dataframe and vocabulary are constructed from the file and returned. The vocabulary file that is
    generated during modification of test input data should be ignored
    """
    if train_flag:
        datafilename = os.path.join(DATA_DIR, 'train_modified.csv')
    else:
        datafilename = os.path.join(DATA_DIR, 'test_modified.csv')
        
    global vocab_dump
    vocabfilename = os.path.join(DATA_DIR, 'vocab.pickle')
    
    if os.path.exists(datafilename) and os.path.exists(vocabfilename):
        df = pd.read_csv(datafilename)
        vocab_op = open(vocabfilename, 'rb')
        vocab_dump = pickle.load(vocab_op)
        vocab_op.close()
    else:
        vocab_dump = Counter()
        df = generate_modified_datafile(input_dataframe, num_threads, vocab_flag=train_flag)
        df.to_csv(datafilename, index=False, encoding='utf-8')
        if train_flag:
            vocab_op = open(vocabfilename, 'wb')
            pickle.dump(vocab_dump, vocab_op)
            vocab_op.close()
            
    return df, vocab_dump





