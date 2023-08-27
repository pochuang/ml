import tensorflow as tf
import numpy as np

with tf.Session() as sess:
  total = tf.Variable(tf.zeros([1, 2]))
  weights = tf.Variable(tf.random_uniform([1,2]))
  #tf.initialize_all_variables().run()
  tf.global_variables_initializer().run()
  update_weights = tf.assign(weights, tf.random_uniform([1, 2], -1.0, 1.0))
  update_total = tf.assign(total, tf.add(total, weights))
  for _ in range(5):
    sess.run(update_weights)
    sess.run(update_total)
   
    print weights.eval(), total.eval()