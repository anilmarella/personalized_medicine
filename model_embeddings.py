# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:22:38 2017

@author: anilm
"""
import os, pickle, math
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from nltk.tokenize import sent_tokenize, word_tokenize

DATA_DIR = 'data'
LOG_DIR = 'embedding_logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
window_size = 2

def write_metadata(data):
    filename = os.path.join(LOG_DIR, 'metadata.tsv')
    with open(filename,'w') as met_file:
#        met_file.write("Word" +"\t"+"Frequency"+"\n")
        for w, f in data:
            met_file.write(w +"\t"+str(f)+"\n")

def create_features_labels_all(train_df, vocabulary):
    result = []
    lim = train_df.shape[0]
    for i in range(0, lim):
        text = train_df.iloc[i]['Text']
        sents = sent_tokenize(text)
        for sent in sents:
            word_idxs = []
            res = []
            for w in word_tokenize(sent):
                word_idxs.append(vocabulary.get(w, -1))
            for i in range(0, len(word_idxs)):
                for j in range(1, window_size+1):
                    if (i-j) >= 0:
                        if word_idxs[i] != -1 and word_idxs[i-j] != -1 and word_idxs[i]!=word_idxs[i-j]:
                            res.append([word_idxs[i], word_idxs[i-j]])
                    if (i+j) < len(word_idxs):
                        if word_idxs[i] != -1 and word_idxs[i+j] != -1 and word_idxs[i]!=word_idxs[i+j]:
                            res.append([word_idxs[i], word_idxs[i+j]])
            result.extend(res)
    return result

def load_features_labels_all(train_df, vocabulary, from_file_flag):
    filename = os.path.join(DATA_DIR, 'embedding_input_labels.pickle')
    output = None
    res = None
    if from_file_flag:
        output = open(filename, 'rb')
        res = pickle.load(output)
    else:
        output = open(filename, 'wb')
        res = create_features_labels_all(train_df, vocabulary)
        pickle.dump(res, output)
    output.close()  
    return res

def train_embeddings_model(train_df, vocab_dump, vocab_size, embedding_size, batch_size):
    write_metadata(vocab_dump.most_common(vocab_size))
    vocabulary = dict()
    for w,_ in vocab_dump.most_common(vocab_size):
        vocabulary[w] = len(vocabulary)
    
    embedding_input_labels =  load_features_labels_all(train_df, vocabulary, from_file_flag=False)
    print("Done making the inputs and labels for embeddings.")
    
    inputs_np = np.array([itm[0] for itm in embedding_input_labels])
    labels_np = np.array([itm[1] for itm in embedding_input_labels])
    print(inputs_np.shape, labels_np.shape)
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None,1])
    all_words_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    tf.summary.histogram('WordEmbeddings',all_words_embeddings)
    hidden_layer = tf.nn.embedding_lookup(all_words_embeddings, train_inputs)
    weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocab_size]))

    output_layer = tf.matmul(hidden_layer, tf.transpose(weights)) + biases
    train_one_hot = tf.one_hot(train_labels, vocab_size)
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=train_one_hot))
    tf.summary.scalar('Xent', xent)

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(xent)
    embedding_var = tf.get_variable('all_words_embeddings', [vocab_size, embedding_size])

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'metadata.tsv'
    
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
    
    #saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        lim = (int(len(embedding_input_labels)/batch_size)+1)*batch_size
        while i < lim:
            inputs = inputs_np[i:i+batch_size]
            labels = labels_np[i:i+batch_size].reshape([-1, 1])
            i += batch_size
            feed_dict = {train_inputs: inputs, train_labels: labels}
            _, xent_val = sess.run([optimizer, xent], feed_dict=feed_dict)
            if i%1000 ==0:
                print("Cross entropy at", str(i)+"th", "iteration is", str(xent_val))
                summ = sess.run(summaries, feed_dict=feed_dict)
                summary_writer.add_summary(summ, i)
        embeds = sess.run(all_words_embeddings)
    return embeds, vocabulary

def train_embeddings(train_df, vocab_dump, vocab_size, embedding_size, batch_size):
    embedding_filename = 'embeddings_dict.pickle'
    if os.path.exists(embedding_filename):
        output = open(embedding_filename, 'rb')
        embeddings_dict = pickle.load(output)
    else:
        embeds, vocabulary = train_embeddings_model(train_df, vocab_dump, vocab_size, embedding_size, batch_size)
        embeddings_dict = dict(zip(vocabulary.keys(), embeds))
        output = open(embedding_filename, 'wb')
        pickle.dump(embeddings_dict, output)
    output.close()
    return embeddings_dict

