import pandas as pd
import tensorflow as tf
import numpy as np
import math, pickle, os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter

train_df = pd.read_csv('train_modified.csv')
train_df.drop_duplicates('Text', inplace=True)
print(train_df.shape)
print(train_df.head())

vocab_size = 500
window_size = 2
embedding_size=20
LOG_DIR = 'logs4'

def write_metadata(data):
    filename = os.path.join(LOG_DIR, 'metadata.tsv')
    with open(filename,'w') as met_file:
        met_file.write("Word" +"\t"+"Frequency"+"\n")
        for w, f in data:
            met_file.write(w +"\t"+str(f)+"\n")

def create_vocabulary(vocab_size):
    stemmer = PorterStemmer()
    vocab_list = Counter()
    vocabulary = dict()
    for i in range(0, train_df.shape[0]):
        for word in word_tokenize(train_df.iloc[i]['Text']):
            if word != '.':
                word = stemmer.stem(word)
                vocab_list[word] += 1
    write_metadata(vocab_list.most_common(vocab_size))
    for w,_ in vocab_list.most_common(vocab_size):
        vocabulary[w] = len(vocabulary)
    return vocabulary

def load_vocabulary():
    vocab_filename = 'dictionary.pickle'
    if not os.path.exists(vocab_filename):
        output = open(vocab_filename, 'wb')
        vocabulary = create_vocabulary(vocab_size)
        pickle.dump(vocabulary, output)
        output.close()
    else:
        output = open(vocab_filename, 'rb')
        vocabulary = pickle.load(output)
        output.close()
        
    return vocabulary

vocabulary = load_vocabulary()

def get_features_labels_sentence(sent):
    word_idxs = []
    for w in word_tokenize(sent):
        word_idxs.append(vocabulary.get(w, -1))
    result = []
    for i in range(0, len(word_idxs)):
        for j in range(1, window_size+1):
            if (i-j) >= 0:
                if word_idxs[i] != -1 and word_idxs[i-j] != -1 and word_idxs[i]!=word_idxs[i-j]:
                    result.append([word_idxs[i], word_idxs[i-j]])
            if (i+j) < len(word_idxs):
                if word_idxs[i] != -1 and word_idxs[i+j] != -1 and word_idxs[i]!=word_idxs[i+j]:
                    result.append([word_idxs[i], word_idxs[i+j]])
    return result

def get_features_lables_text(text):
    sents = sent_tokenize(text)
    result = []
    for i in range(0, len(sents)):
        result.extend(get_features_labels_sentence(sents[i]))
    return result

def get_features_lables_all():
    result = []
    lim = train_df.shape[0]
    for i in range(0, lim):
        result.extend(get_features_lables_text(train_df.iloc[i]['Text']))
    return result


def load_features_labels_all():
    filename = 'input_labels.pickle'
    if not os.path.exists(filename):
        output = open(filename, 'wb')
        res = get_features_lables_all()
        pickle.dump(res, output)
        output.close()
    else:
        output = open(filename, 'rb')
        res = pickle.load(output)
        output.close()
        
    return res


input_labels =  load_features_labels_all()
print("Done making the list.")

ip_list = [itm[0] for itm in input_labels]
lbl_list = [itm[1] for itm in input_labels]
print(len(ip_list), len(lbl_list))

ip_np = np.array(ip_list)
lbl_np = np.array(lbl_list)

num_sampled = 10
batch_size = 100

embeddings = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(
  tf.truncated_normal([vocab_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocab_size))

# Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)


embedding_var = tf.get_variable('embeddings', [vocab_size, embedding_size])

from tensorflow.contrib.tensorboard.plugins import projector
# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join('metadata.tsv')

# Saves a configuration file that TensorBoard will read during startup.
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    lim = (int(len(input_labels)/batch_size)+1)*batch_size
    while i < lim:
        if i%1000 ==0:
            print(str(i)+"th","iteration")
        inputs = ip_np[i:i+batch_size]
        labels = lbl_np[i:i+batch_size].reshape([batch_size, 1])
        i += batch_size
        feed_dict = {train_inputs: inputs, train_labels: labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)
    embeds = sess.run(embeddings)
    print(embeds)










