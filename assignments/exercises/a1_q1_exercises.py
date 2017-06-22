"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.sub(x, y))

with tf.Session() as sess:
    print "Problem 1a:"
    print "x is: %s" %x.eval()
    print "y is: %s" %y.eval()
    print "Problem will add them together if x>y, or subtract them if x<y"
    sess.run(out)
    print out.eval()
    print "*"*10

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform([],-1,1)
y = tf.random_uniform([],-1,1)
add_x_y = lambda: tf.add(x,y)
sub_x_y = lambda: tf.sub(x,y)
return_zero = lambda: tf.zeros([])
check_case = tf.case(pred_fn_pairs=[
                        (tf.less(x,y),add_x_y),
                        (tf.less(y,x),sub_x_y)
                    ],
            default=return_zero,
            exclusive=True)

with tf.Session() as sess:
    print "Problem 1b:"
    print "x is: %s" % x.eval()
    print "y is: %s" % y.eval()
    print "Problem will add them together if x>y, or subtract them if x<y"
    print "OR print return 0 otherwise (if they are equal)"
    sess.run(check_case)
    print "*"*10

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

# YOUR CODE
x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.zeros(dtype=tf.int32, shape=[2,3])
check_xy_equals = tf.equal(x,y)

with tf.Session() as sess:
    print "Problem 1c:"
    print "x is: %s" % x.eval()
    print "y is: %s" % y.eval()
    print "Problem will add them together if x>y, or subtract them if x<y"
    print "but x<3 should always be true, so we should see all 0s"
    sess.run(check_xy_equals)
    print check_xy_equals.eval()
    print "*"*10

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

# YOUR CODE

x=tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
  30.97266006,  26.67541885,  38.08450317,  20.74983215,
  34.94445419,  34.45999146,  29.06485367,  36.01657104,
  27.88236427,  20.56035233,  30.20379066,  29.51215172,
  33.71149445,  28.59134293,  36.05556488,  28.66994858])
indicies = tf.where(tf.less(30.0,x))
over_30 = tf.gather(x, indicies)

with tf.Session() as sess:
    print "Problem 1d:"
    print "Should only print elements from tensor over 30"
    sess.run(over_30)
    print over_30.eval()
    print "*"*10

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

# YOUR CODE
range = tf.range(1,6)
diag = tf.diag(range)

with tf.Session() as sess:
    print "Problem 1e:"
    print "Should make a diagonal tensor 6x6"
    sess.run(diag)
    print diag.eval()
    print "*"*10

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

x = tf.random_normal([10,10])
det = tf.matrix_determinant(x)

with tf.Session() as sess:
    print "Problem 1f:"
    print "Should construct the determinant matrix of a 10x10 normal distributed matrix"
    print "Whatever that means"
    sess.run(det)
    print det.eval()
    print "*"*10

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

# YOUR CODE
x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
uniques, uniques_indicies = tf.unique(x)

with tf.Session() as sess:
    print "Problem 1g:"
    print "x is %s"%x.eval()
    print "Should print the uniques from x"
    sess.run(uniques)
    print uniques.eval()
    print "*"*10

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

# YOUR CODE
x = tf.random_normal([300])
y = tf.random_normal([300])

residual = tf.abs(x-y)
residual_average = tf.metrics.mean(residual)
condition = tf.less(residual_average, 0)#tf.assert_negative #will raise an error
mean_square_error = tf.square(x-y)/len(x)
if_true = lambda: mean_square_error
sum_abs_value = tf.add(tf.abs(x-y))
if_false = sum_abs_value
final_op = tf.cond(condition, if_true, if_false)

with tf.Session() as sess:
    print "Problem 1g:"
    print "x is %s"%x.eval()
    print "y is %s"%y.eval()
    print "Problem should print MSE if residual average is negative"
    print "or sum of absolute value of residual if not"
    sess.run(final_op)
    print final_op.eval()
    print "*"*10
