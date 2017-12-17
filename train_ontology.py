# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:12:41 2017

@author: anilm
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle, os

train_df = pd.read_csv('train_text_modified.csv')
vocab_df = pd.read_csv('vocab/full_vocab.csv')

#vocab_df = vocab_df.iloc[0:200, :]
#gene_dict = {'word':train_df['Gene'].unique(), 'frequency':0}
#df = pd.DataFrame(gene_dict)
#vocab_df = vocab_df.append(df, ignore_index=True)

vocab_size = vocab_df.shape[0]
m = train_df.shape[0]

features = np.ndarray(shape=(m, vocab_size))
labels = train_df['Class'].values.reshape(-1, 1)
output_labels = len(np.unique(labels))
features_filename = 'features.pickle'

for i in range(m):
    if i %500 ==0:
        print('Still in {}th row'.format(i))
    for j in range(vocab_size):
        line_text = train_df.iloc[i]['Text']
        word = vocab_df.iloc[j]['word']
        features[i, j] = line_text.count(word)
        line_text = line_text.replace(word, "UNK")

print("Done creating the features!")
vocab_op = open(features_filename, 'wb')
pickle.dump(features, vocab_op)
vocab_op.close()

#if(os.path.exists(features_filename)):
#    vocab_op = open(features_filename, 'rb')
#    features = pickle.load(vocab_op)
#    vocab_op.close()
#else:
#    print("Sorry run the features code again. Unable to find the file.")


def construct_model(input_X, input_y):
    onehot_y = tf.reshape(tf.one_hot(input_y, output_labels), (-1, 9))
    lambd = 0.001
    hl1_size = 100
#    hl1_size = output_labels
#    hl2_size = 50
    hl2_size = output_labels
    w1 = tf.get_variable("W1", [vocab_size, hl1_size], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [1, hl1_size], initializer=tf.zeros_initializer())
    a1 = tf.add(tf.matmul(input_X , w1), b1)
#    
    w2 = tf.get_variable("W2", [hl1_size, hl2_size], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [1, hl2_size], initializer=tf.zeros_initializer())
#    a2 = tf.add(tf.matmul(a1 , w2), b2)
#    
#    w3 = tf.get_variable("W3", [hl2_size, output_labels], initializer=tf.contrib.layers.xavier_initializer())
#    b3 = tf.get_variable("b3", [1, output_labels], initializer=tf.zeros_initializer())
    
    logits = tf.add(tf.matmul(a1, w2), b2)
#    logits = tf.add(tf.matmul(input_X, w1), b1)
#    logits = tf.add(tf.matmul(a2, w3), b3)
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_y))
    regularizer = tf.nn.l2_loss(w1)
    regularization_penalty = tf.multiply(lambd, regularizer)
    xent = tf.add(xent, regularization_penalty)
    
    is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return xent, accuracy


X_train, X_cv, y_train, y_cv = train_test_split(features, labels, test_size=0.1)

#X_train=features
#y_train=labels

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)

print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)

input_X = tf.placeholder(tf.float32, shape=[None, vocab_size])
input_y = tf.placeholder(tf.int64, shape=[None,1])

xent, accuracy = construct_model(input_X, input_y)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(xent)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 0
    while iteration < 2000:
        feed_dict = {input_X: X_train, input_y: y_train}
        _, xent_entropy, acc = sess.run([train_step, xent, accuracy], feed_dict=feed_dict)
        if iteration % 100 == 0:
            print("Cross entropy at {}th iteration is {} and accuracy is {}".format(iteration, xent_entropy, acc))
        iteration+=1
    test_xent_entropy, test_acc = sess.run([xent, accuracy], feed_dict={input_X: X_cv, input_y: y_cv})
    print("Cross entropy for test is {} and Testing accuracy is: {}".format(test_xent_entropy, test_acc))