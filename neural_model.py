# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:50:55 2017

@author: anilm
"""

#import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

embedding_size = 20
hidden_units_l1 = 50
output_labels = 9

def generate_row_embeddings(df_txt, word_embeddings):
    lim = df_txt.shape[0]
    embeddings = np.ndarray(shape=[lim, embedding_size])
    for i in np.arange(lim):
        row_embedding=np.zeros([embedding_size,])
        sents = sent_tokenize(df_txt.iloc[i])
        for sent in sents:
            words = word_tokenize(sent)
            for word in words:
                if word in word_embeddings:
                    row_embedding += word_embeddings[word]
        embeddings[i] = row_embedding.reshape([1,-1])
    return embeddings

def construct_model(input_X, input_y):
    onehot_y = tf.one_hot(input_y, output_labels)
    w1 = tf.Variable(tf.truncated_normal([embedding_size, hidden_units_l1], stddev=np.sqrt(2./embedding_size)))
    b1 = tf.Variable(tf.constant(0.1 , shape=[1, hidden_units_l1]))
    w2 = tf.Variable(tf.truncated_normal([hidden_units_l1, output_labels], stddev=np.sqrt(1./hidden_units_l1)))
    b2 = tf.Variable(tf.constant(0.1 , shape=[1, output_labels]))
    
    hl1 = tf.nn.relu(tf.matmul(input_X, w1) + b1)
    output = tf.nn.softmax(tf.matmul(hl1, w2) + b2)
   
    is_correct = tf.equal(tf.arg_max(output,1), tf.arg_max(onehot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=onehot_y))
    
    return xent, accuracy

def train_model(train_df, test_df, word_embeddings):
    X = generate_row_embeddings(train_df['Text'], word_embeddings)
    y = train_df['Class'].values -1
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, y_train.shape)
#    X_test = generate_row_embeddings(test_df['Text'], word_embeddings)

    input_X = tf.placeholder(tf.float32, shape=[None, embedding_size])
    input_y = tf.placeholder(tf.int64, shape=[None,])
    xent, accuracy = construct_model(input_X, input_y)
    train_step = tf.train.AdamOptimizer().minimize(xent)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0
        while iteration < 500:
            feed_dict = {input_X: X_train, input_y: y_train}
            _, xent_entropy, acc = sess.run([train_step, xent, accuracy], feed_dict=feed_dict)
            print("Cross entropy at {}th iteration is {} and accuracy is {}".format(iteration, xent_entropy, acc))
            iteration+=1
        test_acc = sess.run(accuracy, feed_dict={input_X: X_cv, input_y: y_cv})
        print("Testing accuracy is: {}".format(test_acc))

def test():
    train_df = pd.read_csv('data/train_modified.csv')
    test_df = pd.read_csv('data/test_modified.csv')
    embedding_filename = 'embeddings_dict.pickle'
    output = open(embedding_filename, 'rb')
    word_embeddings = pickle.load(output)
    output.close()
    train_model(train_df, test_df, word_embeddings)
    
if __name__ == '__main__':
  test()