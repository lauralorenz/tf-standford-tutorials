from utils import define_scope
import io
import numpy as np
import csv
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
            target.append(float(row.pop('chd')))
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
    features = tf.placeholder(dtype=tf.float64, shape=[1,9])
    response = tf.placeholder(dtype=tf.float64, shape=[1])

    # Step 4: create weights and bias
    w = tf.Variable(tf.random_normal(dtype=tf.float64, shape=[9,1], stddev=0.01), name="weights")
    b = tf.Variable(tf.zeros(dtype=tf.float64, shape=[1]), name="bias")

    # Step 5: logistic regression model. predict Y from X and w,b
    logits = features * w + b

    # Step 6: define loss function
    # softmax cross entropy with logits as the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits, response)
    # loss = tf.reduce_mean(entropy) # used for batch case

    # Step 7: define training optimizer
    # use gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(entropy)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # run
        for i in range(n_epoch):
            for x,y in zip(X_train, Y_train):
                reshaped_x = x#x.reshape([1,9])
                reshaped_y = y#y.reshape([1])
                print(reshaped_x)
                print(reshaped_y)
                _, loss = sess.run([optimizer, entropy], feed_dict={features:reshaped_x, response:reshaped_y})
                print(loss)
            print("epoch %s" % i)


        # test

        # get a version of one that is the right data type
        one = tf.ones(shape=[], dtype=tf.int32)
        # run through model with test set and accrue prediction scores
        accuracy_count = 0
        for x,y in zip(X_test, Y_test):
            reshaped_x = x.reshape([1, 9])
            reshaped_y = y.reshape([1, 1])
            # TODO: example pulls optimizer and loss here too, but doesn't that edit the model since the vars are global?
            _, loss, predicted_Y = sess.run([optimizer, entropy, logits], feed_dict={features:reshaped_x, response:reshaped_y})

            pred = tf.nn.softmax(predicted_Y)
            print(pred.eval())
            print(y)
            correct_preds = tf.equal(pred[0][0], y)

            accuracy_count += sess.run(correct_preds)

        print "Accuracy {0}".format(accuracy_count/len(Y_test))

        # draw?
