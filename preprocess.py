# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 00:37:47 2017

@author: anilm
"""

import pandas as pd
import os, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


#manually_chosen = ['powerpoint', 'slide', 'shown', 'fig', 'figure', 'download', 'manuscript', 'com', 'image', 
#                   'please', 'contact', 'figureopen', 'tabdownload', 'page', 'http', 'table', 'bloodjournal', 'org',
#                   'www','discovery', 'october', 'journal', 'database', 'nih', 'gov', 'nhgri', 'institute', 'farber',
#                   'aacrjournals', 'google', 'web', 'scholar', 'science', 'study', 'data', 'article', 'must', 
#                   'therefore', 'hereby', 'author', 'described']

DATA_DIR = 'data'
datafilename = os.path.join(DATA_DIR, 'training_text')
train_text = pd.read_csv(datafilename, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

variantsfilename = os.path.join(DATA_DIR, 'training_variants')
train_variants = pd.read_csv(variantsfilename)

train_df = pd.merge(train_variants, train_text, on='ID')
train_df.drop_duplicates('Text', inplace=True)
del train_text
del train_variants

#stop_words = stopwords.words('English')
#stop_words.extend(manually_chosen)

def check_eligibility(word):
    if not re.match('^\d*[a-zA-Z][a-zA-Z0-9]*$', word):
        return False
    if len(word) <= 2:
        return False
    if word.isnumeric():
        return False
    return True

m = train_df.shape[0]
spl_char_splitter = re.compile('[\/|\-|\.|,]')
lemma = WordNetLemmatizer()
for i in range(0,m):
    sents = sent_tokenize(train_df.iloc[i]['Text'])
    modified_sents = []
    for sent in sents:
        if re.search(spl_char_splitter, sent):
            sent = re.sub(spl_char_splitter, " ", sent)
        words = word_tokenize(sent.lower())
        words = [lemma.lemmatize(w) for w in words]
        words = [w for w in words if check_eligibility(w)]
#        words = [w if w not in stop_words else 'UNK' for w in words]
        modified_sents.append(" ".join(words))
    train_df.iloc[i, train_df.columns.get_loc('Text')] = ". ".join(modified_sents)

# Write to a file.. Just in case
train_df.to_csv('train_text_modified.csv', index=False, encoding='utf-8')
train_df = pd.read_csv('train_text_modified.csv')

"""
# Creating ontology from here on...
window_size = 2
stop_words = stopwords.words('English')

def create_vocab_per_row(row_text):
    sw_row = Counter()
    dw_row = Counter()
    tw_row = Counter()
    escape_sq = ['UNK', '.']
    sents = sent_tokenize(row_text)
    
    for sent in sents:
        words = word_tokenize(sent)
        for i in range(0, len(words)):
            if words[i] not in escape_sq:
                sw_row[words[i]] += 1
                if i+1 < len(words) and words[i+1] not in escape_sq:
                    dw_row[words[i]+" "+words[i+1]] += 1
                    if i+2 < len(words) and words[i+2] not in escape_sq:
                        tw_row[words[i]+" "+words[i+1]+" "+words[i+2]] += 1
                        
    sw_row_df = pd.DataFrame.from_records(sw_row.most_common(), columns=['word','frequency'])                        
    dw_row_df = pd.DataFrame.from_records(dw_row.most_common(), columns=['word','frequency'])                        
    tw_row_df = pd.DataFrame.from_records(tw_row.most_common(), columns=['word','frequency'])                        
    
    return sw_row_df, dw_row_df, tw_row_df

def create_vocab_per_df(input_df):
    sw_all_rows = []
    dw_all_rows = []
    tw_all_rows = []
    frac = 0.9
    m = input_df.shape[0]
    for i in range(0,m):
        row_text = input_df.iloc[i]['Text']
        sw_row_df, dw_row_df, tw_row_df = create_vocab_per_row(row_text)
        sw_all_rows.append(sw_row_df)
        dw_all_rows.append(dw_row_df)
        tw_all_rows.append(tw_row_df)
    
    sw_allrows_df = pd.concat(sw_all_rows).groupby('word', as_index=False).agg({'frequency':['count', 'sum']})
    dw_allrows_df = pd.concat(dw_all_rows).groupby('word', as_index=False).agg({'frequency':['count', 'sum']})
    tw_allrows_df = pd.concat(tw_all_rows).groupby('word', as_index=False).agg({'frequency':['count', 'sum']})
    
    sw_allrows_df.columns = [x if y=="" else y for x,y in sw_allrows_df.columns]
    dw_allrows_df.columns = [x if y=="" else y for x,y in dw_allrows_df.columns]
    tw_allrows_df.columns = [x if y=="" else y for x,y in tw_allrows_df.columns]
    
    sw_df = sw_allrows_df.loc[sw_allrows_df['count']>(frac*m), ['word', 'sum']].sort_values('sum', ascending=False)
    dw_df = dw_allrows_df.loc[dw_allrows_df['count']>(frac*m), ['word', 'sum']].sort_values('sum', ascending=False)
    tw_df = tw_allrows_df.loc[tw_allrows_df['count']>(frac*m), ['word', 'sum']].sort_values('sum', ascending=False)

    return sw_df, dw_df, tw_df

def correct_frequencies(sw_df, dw_df, tw_df):
    grouped_dw = dw_df.groupby('word1', as_index=False).agg({'frequency':'sum'}).sort_values('frequency', ascending=False)
    temp_sw = pd.merge(sw_df, grouped_dw, left_on='word1', right_on='word1', how='left')
    temp_sw['frequency_y'].fillna(0, inplace=True)
    temp_sw.loc[:, 'frequency_x'] = temp_sw.loc[:, 'frequency_x'] - temp_sw.loc[:, 'frequency_y']
    sw_df = temp_sw.loc[:, ['word1','frequency_x']]
    
    grouped_tw = tw_df.groupby(['word1','word2'], as_index=False).agg({'frequency':'sum'}).sort_values('frequency', ascending=False)
    temp_dw = pd.merge(dw_df, grouped_tw, left_on=['word1','word2'], right_on=['word1','word2'], how='left') 
    temp_dw['frequency_y'].fillna(0, inplace=True)
    temp_dw.loc[:, 'frequency_x'] = temp_dw.loc[:, 'frequency_x'] - temp_dw.loc[:, 'frequency_y']
    dw_df = temp_dw.loc[:, ['word1','word2','frequency_x']]
    
    sw_df.columns = ['word', 'frequency']
    dw_df.columns = ['word1', 'word2', 'frequency']
    dw_df['word'] = dw_df['word1'] +" "+dw_df['word2']
    tw_df['word'] = tw_df['word1'] +" "+tw_df['word2'] +" "+ tw_df['word3']
    dw_df.drop(['word1', 'word2'], axis=1, inplace=True)
    tw_df.drop(['word1', 'word2', 'word3'], axis=1, inplace=True)
    
    return sw_df, dw_df, tw_df



#sw,dw,tw = create_vocab_per_df(train_df.loc[train_df['Class']==8, :])

# List of Dataframes for each set of vocabularies.
single_words_dfs = []
double_words_dfs = []
triple_words_dfs = []

for i in range(9):
    label = i+1
    single_words_df, double_words_df, triple_words_df = create_vocab_per_df(train_df.loc[train_df['Class']==label, :])
#    single_words_df, double_words_df, triple_words_df = correct_frequencies(single_words_df, double_words_df, triple_words_df)
    
    single_words_dfs.append(single_words_df)
    double_words_dfs.append(double_words_df)
    triple_words_dfs.append(triple_words_df)

# Merging all lists and aggregating by count
sw_full_groups = pd.concat(single_words_dfs).groupby('word', as_index=False).agg({'sum':'count'})
dw_full_groups = pd.concat(double_words_dfs).groupby('word', as_index=False).agg({'sum':'count'})
tw_full_groups = pd.concat(triple_words_dfs).groupby('word', as_index=False).agg({'sum':'count'})

for idx in range(len(single_words_dfs)):
    df = single_words_dfs[idx]
    single_words_dfs[idx] = df.loc[~df['word'].isin(sw_full_groups.loc[sw_full_groups['sum']>3, 'word']), :]
    single_words_dfs[idx].sort_values('sum', inplace=True, ascending=False)

    df2 = double_words_dfs[idx]
    double_words_dfs[idx] = df2.loc[~df2['word'].isin(dw_full_groups.loc[dw_full_groups['sum']>3, 'word']), :]
    double_words_dfs[idx].sort_values('sum', inplace=True, ascending=False)
    
    df3 = triple_words_dfs[idx]
    triple_words_dfs[idx] = df3.loc[~df3['word'].isin(tw_full_groups.loc[tw_full_groups['sum']>3, 'word']), :]
    triple_words_dfs[idx].sort_values('sum', inplace=True, ascending=False)    

for i in range(9):
    label = i+1
    sw_fn = 'vocab/single_words_vocab_'+str(label)+'.csv'
    single_words_dfs[i].to_csv(sw_fn, index=False, encoding='utf-8')
    dw_fn = 'vocab/double_words_vocab_'+str(label)+'.csv'
    double_words_dfs[i].to_csv(dw_fn, index=False, encoding='utf-8')
    tw_fn = 'vocab/triple_words_vocab_'+str(label)+'.csv'
    triple_words_dfs[i].to_csv(tw_fn, index=False, encoding='utf-8')

del sw_full_groups, dw_full_groups, tw_full_groups

for idx in range(len(single_words_dfs)):
    single_words_dfs[idx] = single_words_dfs[idx].iloc[:, :]
    double_words_dfs[idx] = double_words_dfs[idx].iloc[:, :]
    triple_words_dfs[idx] = triple_words_dfs[idx].iloc[:, :]
    
all_dfs = single_words_dfs + double_words_dfs + triple_words_dfs
full_vocab = pd.concat(all_dfs)
full_vocab.drop_duplicates('word', inplace=True)
print(full_vocab)

full_vocab.to_csv('vocab/full_vocab.csv', index=False, encoding='utf-8')
"""


















