""" The mo frills implementation of word2vec skip-gram model using NCE loss. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# Important Paths
ADMN_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.normpath(os.path.join(ADMN_DIR, "../.."))

# Auto adapt Python path
sys.path.append(BASE_DIR)


import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from examples.process_data import process_data, get_index_vocab

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """
    # Step 1: define the placeholders for input and output
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), 
                            name='embed_matrix')

    # Step 3: define the inference
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    # Step 4: construct variables for NCE loss
    nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                stddev=1.0 / (EMBED_SIZE ** 0.5)), 
                                                name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

    # define loss function to be NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                        biases=nce_bias, 
                                        labels=target_words, 
                                        inputs=embed, 
                                        num_sampled=NUM_SAMPLED, 
                                        num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    with tf.Session() as sess:
        with tf.name_scope("initialize-vars"):
            sess.run(tf.global_variables_initializer())

        with tf.name_scope("train-model"):
            total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('../../my_graph/no_frills/', sess.graph)
            for index in xrange(NUM_TRAIN_STEPS):
                centers, targets = batch_gen.next()
                loss_batch, _ = sess.run([loss, optimizer],
                                        feed_dict={center_words: centers, target_words: targets})
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            writer.close()

def co_occurence():
    WINDOW_SIZE = 4
    DIMS = 10
    # Step 1: Read in data, build vocab
    words, vocab, indicies = get_index_vocab(DIMS)

    # Step 3: Build co-occurence matrix
    with tf.name_scope("build-matrix"):
        print(vocab)
        print(indicies)
        print(vocab["the"])
        print(indicies[1])
        matrix = np.empty([DIMS, DIMS])
        words_processed = 0
        start = 0
        stop = WINDOW_SIZE
        while words_processed <= DIMS:
            target = words[start]
            print("Finding neighbors for {word}".format(word=target))
            if vocab.get(target):
                # we have this word in vocab so it is a valid target
                target_row_index = vocab[target]
            else:
                # move the window up one and find a new target
                start += 1
                stop += 1
                continue
            for word in words[start:stop]:
                word_index = vocab.get(word)
                if not word_index:
                    # we didn't get this word when processing vocab
                    # skip
                    continue
                print("Found neighbor {context_word}".format(context_word=indicies[word_index]))
                matrix[target_row_index][word_index] += 1
            print("Cooccurence for {word} is {matrix}".format(word=target, matrix=matrix[target_row_index]))
            start += 1
            stop += 1
            words_processed += 1

        print(indicies)
        print(words[:stop])
        print(matrix)

    #Step 3: Use SVD to reduce dimensionality of co-occurence matrix to embedding size.
    # Use tf.svd

    with tf.name_scope("reduce-dimensionality"):
        embedding = tf.svd(matrix)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('../../my_graph/countembeddings/', sess.graph)
            s,u,v=sess.run(embedding)
            print(s)

            writer.close()

def main():
    WORD2VEC = True
    if WORD2VEC:
        batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
        word2vec(batch_gen)
    else:
        co_occurence()

if __name__ == '__main__':
    main()