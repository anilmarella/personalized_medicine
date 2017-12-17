# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:24:21 2017

@author: anilm
"""
# Import necessary libraries
import pandas as pd
import time
from text_modification import fetch_modified_dataframe
from model_embeddings import train_embeddings
from neural_model import train_model

# Constants we want to use
DATA_DIR = 'data'

num_threads_to_use = 2
# Global variables
start = time.time()

train_text = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv('data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_variants = pd.read_csv('data/training_variants')
test_variants = pd.read_csv('data/test_variants')

train_df = pd.merge(train_variants, train_text, on='ID')
train_df.drop_duplicates('Text', inplace=True)
test_df = pd.merge(test_variants, test_text, on='ID')

print('Starting making modified dataframe')
train_modified, vocab_dump = fetch_modified_dataframe(train_df, True, num_threads_to_use)
test_modified, _ = fetch_modified_dataframe(test_df, False,  num_threads_to_use)
print('Phew! finished making modified dataframe','Time taken to do this is',str(time.time() - start))
print('Check data folder for train_modified, test_modified csv files and vocabulary pickle file')

# Save memory
del train_text
del test_text
del train_variants
del test_variants
del train_df
del test_df

vocab_size = 200
embedding_size = 20
batch_size = 1000
print('Starting training word embeddings with batches of',str(batch_size)+".", 'Vocabulary size is',vocab_size, "Embeddings size will be",embedding_size)
embeddings_dict = train_embeddings(train_modified, vocab_dump, vocab_size, embedding_size, batch_size)
print('Finished creating the embeddings dictionary')
print(embeddings_dict)

print('Strating training the neural network model')
train_model(train_modified, test_modified, embeddings_dict)
print('Finished training the neural network model')
stop = time.time()
print("Time taken to nail this thing is: ", str(stop-start))
