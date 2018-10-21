import tensorflow as tf

#flow graph
a = tf.constant(20, name="a")
b = tf.constant(30, name="b")

mul_op = a*b

#session
sess = tf.Session()

# TensorBoard use
tw = tf.summary.FileWriter('log_dir', graph=sess.graph)

# Session run
print(sess.run(mul_op))