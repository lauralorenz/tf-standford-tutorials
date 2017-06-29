from utils import define_scope
import io
import numpy as np
import csv
#import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def read_data():
    FAMHIST_LOOKUP = {"Absent": 0.0, "Present": 1.0}

    data = []
    target = []
    with io.open('../data/heart.csv', 'tr') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["famhist"] = FAMHIST_LOOKUP[row['famhist']]
            target.append(row.pop('chd'))
            data.append([float(row[k]) for k in row.keys()])
    data = np.asarray(data)
    target = np.asarray(target)
    return train_test_split(data, target, test_size=0.10, random_state=42)

if __name__ == '__main__':

    # Step 1: Read in data
    X_train, X_test, Y_train, Y_test = read_data()

    # Step 2: Define parameters for the model
    learning_rate = 0.01
    n_epoch = 10

    # Step 3: Set up placeholders for the features and labels
    features = tf.placeholder(dtype=tf.float64, shape=[9,1])
    response = tf.placeholder(dtype=tf.float64, shape=[])

    # Step 4: create weights and bias
    w = tf.Variable(tf.random_normal(dtype=tf.float64, shape=[1,9], stddev=0.01), name="weights")
    b = tf.Variable(tf.zeros(dtype=tf.float64, shape=[]), name="bias")

    # Step 5: logistic regression model. predict Y from X and w,b
    logits = tf.matmul(features, w) + b

    # Step 6: define loss function
    # softmax cross entropy with logits as the loss function
    #entropy = tf.nn.softmax_cross_entropy_with_logits(logits, response)
    # loss = tf.reduce_mean(entropy) # used for batch case

    # Step 7: define training optimizer
    # use gradient descent
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(entropy)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # run
        for i in range(n_epoch):
            for x,y in zip(X_train, Y_train):
                logits = sess.run(logits, feed_dict={features:x, response:y})
                print logits
                #sess.run(optimizer, feed_dict={features:x, response:y})

        # test
        all_logits = []
        for x,y in zip(X_test, Y_test):
            logits = sess.run(logits, feed_dict={features:x, response:y})
            all_logits.append(x)

        preds = tf.nn.softmax(all_logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_test, 1))
        accuracy = tf.reduce_mean(sum(tf.cast(correct_preds, tf.float32)))
        accuracy = sess.run(accuracy)

        print "Accuracy {0}".format(accuracy)

        # draw?
