import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

addit = tf.add(a,b)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
print "a + b is %i" %sess.run(addit, feed_dict={a: 1, b: 2})

sess.close()